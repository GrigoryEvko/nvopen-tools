// Function: sub_16C7BA0
// Address: 0x16c7ba0
//
__int64 *__fastcall sub_16C7BA0(__pid_t *a1, unsigned int a2, char a3, __int64 **a4)
{
  __pid_t v5; // r13d
  int v6; // r12d
  char v7; // bl
  int v8; // eax
  __pid_t v9; // eax
  __pid_t v10; // r12d
  int v12; // edi
  __m128i *v13; // rax
  __m128i si128; // xmm0
  __int64 *v15; // rdi
  char *v16; // r12
  size_t v17; // rax
  __int64 *v18; // rdi
  size_t v19; // rcx
  __int64 *v20; // rdx
  __int64 *v21; // rsi
  __m128i *v22; // rax
  int v23; // edx
  size_t v24; // rdx
  unsigned int seconds; // [rsp+Ch] [rbp-1C4h]
  int stat_loc; // [rsp+24h] [rbp-1ACh] BYREF
  size_t v28; // [rsp+28h] [rbp-1A8h] BYREF
  __int64 *v29; // [rsp+34h] [rbp-19Ch] BYREF
  int v30; // [rsp+3Ch] [rbp-194h]
  __int64 *v31; // [rsp+40h] [rbp-190h] BYREF
  size_t n; // [rsp+48h] [rbp-188h]
  _QWORD src[2]; // [rsp+50h] [rbp-180h] BYREF
  struct sigaction v34; // [rsp+60h] [rbp-170h] BYREF
  struct sigaction act; // [rsp+100h] [rbp-D0h] BYREF

  seconds = a2;
  v5 = *a1;
  if ( a3 )
  {
    seconds = 0;
    v6 = 0;
  }
  else
  {
    v6 = 1;
    if ( a2 )
    {
      memset(&v34.sa_mask, 0, 0x90u);
      v34.sa_handler = (__sighandler_t)nullsub_617;
      sigemptyset(&v34.sa_mask);
      v6 = 0;
      sigaction(14, &v34, &act);
      alarm(a2);
    }
  }
  v7 = a3 ^ 1;
  sub_16C7610(&v29);
  do
  {
    v9 = waitpid(v5, &stat_loc, v6);
    LODWORD(v29) = v9;
    if ( v9 != -1 || v7 )
    {
      v10 = *a1;
      if ( v9 == *a1 )
      {
        if ( seconds && v7 )
        {
          alarm(0);
          sigaction(14, &act, 0);
        }
        goto LABEL_16;
      }
      if ( !v9 )
      {
LABEL_10:
        v31 = v29;
        LODWORD(n) = v30;
        return v31;
      }
      v8 = *__errno_location();
      if ( !seconds )
        goto LABEL_30;
      if ( v8 == 4 )
      {
        kill(v10, 9);
        alarm(0);
        sigaction(14, &act, 0);
        if ( wait((__WAIT_STATUS)&stat_loc) == v5 )
        {
          v23 = 0;
          v31 = src;
          strcpy((char *)src, "Child timed out");
          n = 15;
        }
        else
        {
          v28 = 32;
          v31 = src;
          v22 = (__m128i *)sub_22409D0(&v31, &v28, 0);
          v31 = (__int64 *)v22;
          src[0] = v28;
          *v22 = _mm_load_si128((const __m128i *)&xmmword_3F673E0);
          v22[1] = _mm_load_si128((const __m128i *)&xmmword_3F673F0);
          n = v28;
          *((_BYTE *)v31 + v28) = 0;
          v23 = -1;
        }
        sub_16C6DC0(a4, (__int64)&v31, v23);
        if ( v31 != src )
          j_j___libc_free_0(v31, src[0] + 1LL);
        LODWORD(n) = -2;
        return v29;
      }
LABEL_23:
      v28 = 31;
      v31 = src;
      v13 = (__m128i *)sub_22409D0(&v31, &v28, 0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_42AF0C0);
      v31 = (__int64 *)v13;
      src[0] = v28;
      qmemcpy(&v13[1], "r child process", 15);
      *v13 = si128;
      n = v28;
      *((_BYTE *)v31 + v28) = 0;
      sub_16C6DC0(a4, (__int64)&v31, -1);
      v15 = v31;
      if ( v31 == src )
        goto LABEL_25;
LABEL_24:
      j_j___libc_free_0(v15, src[0] + 1LL);
      goto LABEL_25;
    }
    v8 = *__errno_location();
  }
  while ( v8 == 4 );
  if ( *a1 != -1 )
  {
    if ( seconds )
      goto LABEL_23;
LABEL_30:
    if ( v8 != 4 )
      goto LABEL_23;
  }
LABEL_16:
  v12 = stat_loc & 0x7F;
  if ( (stat_loc & 0x7F) != 0 )
  {
    if ( (char)(v12 + 1) > 1 )
    {
      if ( a4 )
      {
        v16 = strsignal(v12);
        v17 = strlen(v16);
        sub_2241130(a4, 0, a4[1], v16, v17);
        if ( (stat_loc & 0x80u) != 0 )
        {
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)a4[1]) <= 0xD )
            sub_4262D8((__int64)"basic_string::append");
          sub_2241490(a4, " (core dumped)", 14, a4);
        }
      }
      v30 = -2;
    }
    goto LABEL_10;
  }
  v30 = BYTE1(stat_loc);
  if ( BYTE1(stat_loc) == 127 )
  {
    if ( !a4 )
      goto LABEL_25;
    sub_16F1490(&v31, 2);
    v18 = *a4;
    if ( v31 == src )
    {
      v24 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)v18 = src[0];
        else
          memcpy(v18, src, n);
        v24 = n;
        v18 = *a4;
      }
      a4[1] = (__int64 *)v24;
      *((_BYTE *)v18 + v24) = 0;
      v18 = v31;
    }
    else
    {
      v19 = n;
      v20 = (__int64 *)src[0];
      if ( v18 == (__int64 *)(a4 + 2) )
      {
        *a4 = v31;
        a4[1] = (__int64 *)v19;
        a4[2] = v20;
      }
      else
      {
        v21 = a4[2];
        *a4 = v31;
        a4[1] = (__int64 *)v19;
        a4[2] = v20;
        if ( v18 )
        {
          v31 = v18;
          src[0] = v21;
          goto LABEL_37;
        }
      }
      v31 = src;
      v18 = src;
    }
LABEL_37:
    n = 0;
    *(_BYTE *)v18 = 0;
    v15 = v31;
    if ( v31 == src )
      goto LABEL_25;
    goto LABEL_24;
  }
  if ( BYTE1(stat_loc) != 126 )
    goto LABEL_10;
  if ( a4 )
    sub_2241130(a4, 0, a4[1], "Program could not be executed", 29);
LABEL_25:
  LODWORD(n) = -1;
  return v29;
}
