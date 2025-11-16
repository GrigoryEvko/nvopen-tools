// Function: sub_C87260
// Address: 0xc87260
//
__int64 __fastcall sub_C87260(__pid_t *a1, __int64 a2, _QWORD *a3, __int64 a4, char a5)
{
  __pid_t v5; // ebx
  char v6; // r15
  _BOOL4 v7; // r14d
  __pid_t v8; // eax
  __pid_t v9; // r13d
  __suseconds_t v11; // rdx
  __suseconds_t v12; // rax
  __int64 ru_maxrss; // rcx
  int v14; // edi
  __int64 v15; // rdi
  char *v16; // r12
  size_t v17; // rax
  _BYTE *v18; // rdi
  __int64 v19; // rdx
  size_t v20; // rcx
  __int64 v21; // rsi
  int v22; // eax
  __m128i *v23; // rax
  int v24; // edx
  size_t v25; // rdx
  char v27; // [rsp+6h] [rbp-27Ah]
  char stat_loc[16]; // [rsp+44h] [rbp-23Ch] BYREF
  __int64 v31; // [rsp+54h] [rbp-22Ch] BYREF
  int v32; // [rsp+5Ch] [rbp-224h]
  _QWORD *m128i_i64; // [rsp+60h] [rbp-220h] BYREF
  size_t n; // [rsp+68h] [rbp-218h]
  _QWORD src[2]; // [rsp+70h] [rbp-210h] BYREF
  struct rusage usage; // [rsp+80h] [rbp-200h] BYREF
  struct sigaction act; // [rsp+110h] [rbp-170h] BYREF
  struct sigaction oact; // [rsp+1B0h] [rbp-D0h] BYREF

  v5 = *a1;
  if ( BYTE4(a2) )
  {
    v6 = BYTE4(a2);
    v7 = a2 == 0;
    memset(&act.sa_mask, 0, 0x90u);
    act.sa_handler = (__sighandler_t)nullsub_163;
    sigemptyset(&act.sa_mask);
    sigaction(14, &act, &oact);
    alarm(a2);
    v27 = 0;
  }
  else
  {
    v27 = 1;
    v6 = 0;
    v7 = 0;
  }
  *(_DWORD *)stat_loc = 0;
  sub_C86E50(&v31);
  if ( a4 && *(_BYTE *)(a4 + 24) )
    *(_BYTE *)(a4 + 24) = 0;
  do
  {
    v8 = wait4(v5, (__WAIT_STATUS)stat_loc, v7, &usage);
    LODWORD(v31) = v8;
    if ( v8 != -1 || v6 )
    {
      v9 = *a1;
      if ( v8 != *a1 )
      {
        if ( !v8 )
        {
LABEL_12:
          m128i_i64 = (_QWORD *)v31;
          LODWORD(n) = v32;
          return (__int64)m128i_i64;
        }
        v22 = *__errno_location();
        if ( BYTE4(a2) )
        {
          if ( v22 == 4 )
          {
            if ( !a5 )
            {
              kill(v9, 9);
              alarm(0);
              sigaction(14, &oact, 0);
              if ( wait((__WAIT_STATUS)stat_loc) == v5 )
              {
                v24 = 0;
                m128i_i64 = src;
                strcpy((char *)src, "Child timed out");
                n = 15;
              }
              else
              {
                *(_QWORD *)&stat_loc[4] = 32;
                m128i_i64 = src;
                v23 = (__m128i *)sub_22409D0(&m128i_i64, &stat_loc[4], 0);
                m128i_i64 = v23->m128i_i64;
                src[0] = *(_QWORD *)&stat_loc[4];
                *v23 = _mm_load_si128((const __m128i *)&xmmword_3F673E0);
                v23[1] = _mm_load_si128((const __m128i *)&xmmword_3F673F0);
                n = *(_QWORD *)&stat_loc[4];
                *((_BYTE *)m128i_i64 + *(_QWORD *)&stat_loc[4]) = 0;
                v24 = -1;
              }
              sub_C86680(a3, (__int64)&m128i_i64, v24);
              if ( m128i_i64 != src )
                j_j___libc_free_0(m128i_i64, src[0] + 1LL);
              LODWORD(n) = -2;
              return v31;
            }
LABEL_16:
            if ( !v27 )
            {
              alarm(0);
              sigaction(14, &oact, 0);
            }
            goto LABEL_18;
          }
        }
        else if ( v22 == 4 )
        {
          goto LABEL_18;
        }
LABEL_28:
        m128i_i64 = src;
        sub_C865D0((__int64 *)&m128i_i64, "Error waiting for child process", (__int64)"");
        sub_C86680(a3, (__int64)&m128i_i64, -1);
        v15 = (__int64)m128i_i64;
        if ( m128i_i64 == src )
          goto LABEL_26;
        goto LABEL_29;
      }
      if ( !BYTE4(a2) )
        goto LABEL_18;
      goto LABEL_16;
    }
  }
  while ( *__errno_location() == 4 );
  if ( *a1 != -1 )
    goto LABEL_28;
LABEL_18:
  if ( a4 )
  {
    v11 = usage.ru_utime.tv_usec + 1000000 * usage.ru_utime.tv_sec;
    v12 = v11 + usage.ru_stime.tv_usec + 1000000 * usage.ru_stime.tv_sec;
    ru_maxrss = usage.ru_maxrss;
    if ( *(_BYTE *)(a4 + 24) )
    {
      *(_QWORD *)a4 = v12;
      *(_QWORD *)(a4 + 8) = v11;
      *(_QWORD *)(a4 + 16) = ru_maxrss;
    }
    else
    {
      *(_QWORD *)a4 = v12;
      *(_QWORD *)(a4 + 8) = v11;
      *(_QWORD *)(a4 + 16) = ru_maxrss;
      *(_BYTE *)(a4 + 24) = 1;
    }
  }
  v14 = stat_loc[0] & 0x7F;
  if ( (stat_loc[0] & 0x7F) != 0 )
  {
    if ( (char)(v14 + 1) > 1 )
    {
      if ( a3 )
      {
        v16 = strsignal(v14);
        v17 = strlen(v16);
        sub_2241130(a3, 0, a3[1], v16, v17);
        if ( stat_loc[0] < 0 )
        {
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a3[1]) <= 0xD )
            sub_4262D8((__int64)"basic_string::append");
          sub_2241490(a3, " (core dumped)", 14, a3);
        }
      }
      v32 = -2;
    }
    goto LABEL_12;
  }
  v32 = (unsigned __int8)stat_loc[1];
  if ( stat_loc[1] == 127 )
  {
    if ( !a3 )
      goto LABEL_26;
    sub_F03820(&m128i_i64, 2);
    v18 = (_BYTE *)*a3;
    if ( m128i_i64 == src )
    {
      v25 = n;
      if ( n )
      {
        if ( n == 1 )
          *v18 = src[0];
        else
          memcpy(v18, src, n);
        v25 = n;
        v18 = (_BYTE *)*a3;
      }
      a3[1] = v25;
      v18[v25] = 0;
      v18 = m128i_i64;
    }
    else
    {
      v19 = src[0];
      v20 = n;
      if ( v18 == (_BYTE *)(a3 + 2) )
      {
        *a3 = m128i_i64;
        a3[1] = v20;
        a3[2] = v19;
      }
      else
      {
        v21 = a3[2];
        *a3 = m128i_i64;
        a3[1] = v20;
        a3[2] = v19;
        if ( v18 )
        {
          m128i_i64 = v18;
          src[0] = v21;
          goto LABEL_40;
        }
      }
      m128i_i64 = src;
      v18 = src;
    }
LABEL_40:
    n = 0;
    *v18 = 0;
    v15 = (__int64)m128i_i64;
    if ( m128i_i64 == src )
      goto LABEL_26;
LABEL_29:
    j_j___libc_free_0(v15, src[0] + 1LL);
    goto LABEL_26;
  }
  if ( stat_loc[1] != 126 )
    goto LABEL_12;
  if ( a3 )
    sub_2241130(a3, 0, a3[1], "Program could not be executed", 29);
LABEL_26:
  LODWORD(n) = -1;
  return v31;
}
