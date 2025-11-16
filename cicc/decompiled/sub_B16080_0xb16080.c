// Function: sub_B16080
// Address: 0xb16080
//
void __fastcall sub_B16080(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int8 *a4)
{
  _BYTE *v6; // r13
  int v7; // edi
  _BYTE *v8; // rax
  __int64 v9; // rdx
  _BYTE *v10; // rsi
  __int64 v11; // r8
  _BYTE *v12; // rdi
  __int64 v13; // rdx
  size_t v14; // rcx
  __int64 v15; // rsi
  const char *v16; // r12
  size_t v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  size_t v20; // r8
  size_t v21; // rdx
  _BYTE *v22; // rdi
  __int64 v23; // rax
  size_t v24; // rdx
  size_t n[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD src[10]; // [rsp+10h] [rbp-50h] BYREF

  v6 = (_BYTE *)(a1 + 48);
  *(_QWORD *)a1 = a1 + 16;
  sub_B14B30((__int64 *)a1, a2, (__int64)&a2[a3]);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  v7 = *a4;
  *(_BYTE *)(a1 + 48) = 0;
  if ( (_BYTE)v7 )
  {
    if ( (unsigned __int8)v7 <= 0x1Cu )
      goto LABEL_16;
    a2 = a4 + 48;
    sub_B157E0((__int64)n, (_QWORD *)a4 + 6);
    *(__m128i *)(a1 + 64) = _mm_loadu_si128((const __m128i *)n);
  }
  else
  {
    a2 = (_BYTE *)sub_B92180(a4);
    if ( a2 )
    {
      sub_B15890(n, (__int64)a2);
      v7 = *a4;
      *(__m128i *)(a1 + 64) = _mm_loadu_si128((const __m128i *)n);
      if ( (_BYTE)v7 == 22 )
        goto LABEL_4;
      goto LABEL_17;
    }
  }
  v7 = *a4;
LABEL_16:
  if ( (_BYTE)v7 == 22 )
    goto LABEL_4;
LABEL_17:
  if ( (unsigned __int8)v7 <= 3u )
  {
LABEL_4:
    v8 = (_BYTE *)sub_BD5D20(a4);
    v10 = v8;
    v11 = (__int64)v8;
    if ( v9 )
    {
      v11 = (__int64)&v8[v9];
      v10 = &v8[*v8 == 1];
    }
    n[0] = (size_t)src;
    sub_B14B30((__int64 *)n, v10, v11);
    v12 = *(_BYTE **)(a1 + 32);
    if ( (_QWORD *)n[0] == src )
    {
      v21 = n[1];
      if ( n[1] )
      {
        if ( n[1] == 1 )
          *v12 = src[0];
        else
          memcpy(v12, src, n[1]);
        v21 = n[1];
        v12 = *(_BYTE **)(a1 + 32);
      }
      *(_QWORD *)(a1 + 40) = v21;
      v12[v21] = 0;
      v12 = (_BYTE *)n[0];
      goto LABEL_10;
    }
    v13 = src[0];
    v14 = n[1];
    if ( v6 == v12 )
    {
      *(_QWORD *)(a1 + 32) = n[0];
      *(_QWORD *)(a1 + 40) = v14;
      *(_QWORD *)(a1 + 48) = v13;
    }
    else
    {
      v15 = *(_QWORD *)(a1 + 48);
      *(_QWORD *)(a1 + 32) = n[0];
      *(_QWORD *)(a1 + 40) = v14;
      *(_QWORD *)(a1 + 48) = v13;
      if ( v12 )
      {
        n[0] = (size_t)v12;
        src[0] = v15;
        goto LABEL_10;
      }
    }
    n[0] = (size_t)src;
    v12 = src;
LABEL_10:
    n[1] = 0;
    *v12 = 0;
    if ( (_QWORD *)n[0] != src )
      j_j___libc_free_0(n[0], src[0] + 1LL);
    return;
  }
  if ( (unsigned __int8)v7 <= 0x15u )
  {
    src[3] = 0x100000000LL;
    src[4] = a1 + 32;
    n[1] = 0;
    n[0] = (size_t)&unk_49DD210;
    memset(src, 0, 24);
    sub_CB5980(n, 0, 0, 0);
    sub_A5BF40(a4, (__int64)n, 0, 0);
    n[0] = (size_t)&unk_49DD210;
    sub_CB5840(n);
    return;
  }
  if ( (unsigned __int8)v7 > 0x1Cu )
  {
    v16 = (const char *)sub_B458E0((unsigned int)(v7 - 29));
    v17 = strlen(v16);
    v18 = *(_QWORD *)(a1 + 40);
    v19 = (__int64)v16;
    v20 = v17;
LABEL_21:
    sub_2241130(a1 + 32, 0, v18, v19, v20);
    return;
  }
  if ( (_BYTE)v7 == 24 )
  {
    v22 = (_BYTE *)*((_QWORD *)a4 + 3);
    if ( !*v22 )
    {
      v23 = sub_B91420(v22, a2);
      v20 = v24;
      v19 = v23;
      v18 = *(_QWORD *)(a1 + 40);
      goto LABEL_21;
    }
  }
}
