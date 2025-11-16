// Function: sub_CABD70
// Address: 0xcabd70
//
__int64 __fastcall sub_CABD70(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  unsigned __int64 v6; // rcx
  unsigned int v7; // r12d
  unsigned __int64 v8; // rax
  char *v9; // r14
  int v10; // ebx
  char *v11; // rax
  char v12; // al
  char *v13; // rsi
  char *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdx
  unsigned __int64 v18; // rbx
  __m128i v19; // xmm0
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // rdi
  char *v27; // rax
  __int64 v28; // rbx
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  int v33; // [rsp+4h] [rbp-7Ch]
  unsigned __int64 v34; // [rsp+8h] [rbp-78h]
  const char *v35; // [rsp+10h] [rbp-70h] BYREF
  __m128i v36; // [rsp+18h] [rbp-68h] BYREF
  _BYTE *v37; // [rsp+28h] [rbp-58h]
  __int64 v38; // [rsp+30h] [rbp-50h]
  _QWORD v39[9]; // [rsp+38h] [rbp-48h] BYREF

  v5 = a1;
  v6 = *(_QWORD *)(a1 + 40);
  v33 = *(_DWORD *)(a1 + 60);
  v34 = v6;
  v7 = *(_DWORD *)(a1 + 56) + 1;
  v8 = *(_QWORD *)(a1 + 48);
  if ( v6 == v8 )
  {
    v35 = "Got empty plain scalar";
    LOWORD(v38) = 259;
LABEL_48:
    v34 = v8 - 1;
    goto LABEL_42;
  }
  v9 = *(char **)(a1 + 40);
  v10 = 0;
  do
  {
    if ( *v9 == 35 )
      break;
    while ( *(char **)(v5 + 48) != v9 )
    {
      if ( *v9 == 58 || !(unsigned __int8)sub_CA7FB0(v5, v9) )
      {
        v9 = *(char **)(v5 + 40);
        if ( *v9 != 58 )
          break;
        if ( !(unsigned __int8)sub_CA7FB0(v5, v9 + 1) )
        {
          v9 = *(char **)(v5 + 40);
          break;
        }
      }
      v11 = sub_CA6050(v5, *(char **)(v5 + 40));
      v9 = v11;
      if ( *(char **)(v5 + 40) == v11 )
        break;
      ++*(_DWORD *)(v5 + 60);
      *(_QWORD *)(v5 + 40) = v11;
    }
    a2 = v9;
    a1 = v5;
    v12 = sub_CA7F80(v5, v9);
    v9 = *(char **)(v5 + 40);
    if ( !v12 )
      break;
    while ( 1 )
    {
      a2 = v9;
      a1 = v5;
      if ( !(unsigned __int8)sub_CA7F80(v5, v9) )
        break;
      while ( 1 )
      {
        v13 = v9;
        v14 = sub_CA6100(v5, v9);
        if ( v9 == v14 )
          break;
        if ( v10 && v7 > *(_DWORD *)(v5 + 60) && *v9 == 9 )
        {
          v28 = *(_QWORD *)(v5 + 336);
          v35 = "Found invalid tab character in indentation";
          v29 = *(_QWORD *)(v5 + 48);
          LOWORD(v38) = 259;
          if ( (unsigned __int64)v9 >= v29 )
            v9 = (char *)(v29 - 1);
          if ( v28 )
          {
            v30 = sub_2241E50(v5, v13, v29 - 1, v15, v16);
            *(_DWORD *)v28 = 22;
            *(_QWORD *)(v28 + 8) = v30;
          }
          if ( !*(_BYTE *)(v5 + 75) )
            sub_C91CB0(*(__int64 **)v5, (unsigned __int64)v9, 0, (__int64)&v35, 0, 0, 0, 0, *(_BYTE *)(v5 + 76));
          goto LABEL_39;
        }
        ++*(_DWORD *)(v5 + 60);
        v9 = v14;
        a1 = v5;
        a2 = v14;
        if ( !(unsigned __int8)sub_CA7F80(v5, v14) )
          goto LABEL_19;
      }
      v27 = sub_CA7C80(v5, v9);
      *(_DWORD *)(v5 + 60) = 0;
      if ( !v10 )
        v10 = 1;
      v9 = v27;
      ++*(_DWORD *)(v5 + 64);
    }
LABEL_19:
    if ( !*(_DWORD *)(v5 + 68) && *(_DWORD *)(v5 + 60) < v7 )
    {
      v9 = *(char **)(v5 + 40);
      break;
    }
    *(_QWORD *)(v5 + 40) = v9;
  }
  while ( *(char **)(v5 + 48) != v9 );
  if ( (char *)v34 != v9 )
  {
    v17 = *(_QWORD *)(v5 + 80);
    *(_QWORD *)(v5 + 160) += 72LL;
    v36.m128i_i64[0] = v34;
    v18 = (v17 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    v37 = v39;
    v38 = 0;
    LOBYTE(v39[0]) = 0;
    LODWORD(v35) = 18;
    v36.m128i_i64[1] = (__int64)&v9[-v34];
    if ( *(_QWORD *)(v5 + 88) >= v18 + 72 && v17 )
    {
      *(_QWORD *)(v5 + 80) = v18 + 72;
      if ( !v18 )
      {
        MEMORY[8] = v5 + 176;
        BUG();
      }
    }
    else
    {
      v18 = sub_9D1E70(v5 + 80, 72, 72, 4);
    }
    *(_QWORD *)v18 = 0;
    *(_QWORD *)(v18 + 8) = 0;
    *(_DWORD *)(v18 + 16) = (_DWORD)v35;
    v19 = _mm_loadu_si128(&v36);
    *(_QWORD *)(v18 + 40) = v18 + 56;
    *(__m128i *)(v18 + 24) = v19;
    sub_CA64F0((__int64 *)(v18 + 40), v37, (__int64)&v37[v38]);
    v20 = *(_QWORD *)v18;
    v21 = *(_QWORD *)(v5 + 176);
    *(_QWORD *)(v18 + 8) = v5 + 176;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v18 = v21 | v20 & 7;
    *(_QWORD *)(v21 + 8) = v18;
    v22 = *(_QWORD *)(v5 + 176) & 7LL | v18;
    *(_QWORD *)(v5 + 176) = v22;
    sub_CA80E0(v5, v22 & 0xFFFFFFFFFFFFFFF8LL, v33, 0, v23, v24);
    v25 = v37;
    *(_WORD *)(v5 + 73) = 0;
    if ( v25 != v39 )
      j_j___libc_free_0(v25, v39[0] + 1LL);
    return 1;
  }
  v8 = *(_QWORD *)(v5 + 48);
  v35 = "Got empty plain scalar";
  LOWORD(v38) = 259;
  if ( v34 >= v8 )
    goto LABEL_48;
LABEL_42:
  v31 = *(_QWORD *)(v5 + 336);
  if ( v31 )
  {
    v32 = sub_2241E50(a1, a2, "Got empty plain scalar", v6, a5);
    *(_DWORD *)v31 = 22;
    *(_QWORD *)(v31 + 8) = v32;
  }
  if ( !*(_BYTE *)(v5 + 75) )
    sub_C91CB0(*(__int64 **)v5, v34, 0, (__int64)&v35, 0, 0, 0, 0, *(_BYTE *)(v5 + 76));
LABEL_39:
  *(_BYTE *)(v5 + 75) = 1;
  return 0;
}
