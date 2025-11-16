// Function: sub_34D1730
// Address: 0x34d1730
//
unsigned __int64 __fastcall sub_34D1730(__int64 a1, __int64 *a2, int a3, unsigned int a4, __int64 a5)
{
  int v7; // ebx
  __int64 v8; // r13
  __int64 v9; // r8
  int v10; // r15d
  unsigned __int64 v11; // rbx
  unsigned int i; // ecx
  unsigned __int64 v13; // rax
  __int64 *v14; // rsi
  unsigned int v15; // eax
  bool v16; // of
  int v17; // r15d
  __int64 v18; // rdx
  unsigned int j; // ecx
  __int64 v20; // rax
  __int64 *v21; // rsi
  unsigned int v22; // eax
  unsigned __int64 v23; // r12
  unsigned int v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  unsigned int v29; // [rsp+18h] [rbp-48h]
  unsigned __int64 v30; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+28h] [rbp-38h]

  v7 = a4 * a3;
  v27 = sub_BCDA70(a2, a4);
  v8 = sub_BCDA70(a2, v7);
  sub_C4DEC0((__int64)&v30, a5, a4, 0);
  v9 = v27;
  if ( *(_BYTE *)(v27 + 8) == 18 )
  {
    v11 = 0;
    if ( *(_BYTE *)(v8 + 8) == 18 )
    {
LABEL_22:
      v23 = v11;
      goto LABEL_23;
    }
  }
  else
  {
    v10 = *(_DWORD *)(v27 + 32);
    v11 = 0;
    if ( v10 > 0 )
    {
      for ( i = 0; i != v10; ++i )
      {
        v13 = v30;
        if ( v31 > 0x40 )
          v13 = *(_QWORD *)(v30 + 8LL * (i >> 6));
        if ( (v13 & (1LL << i)) != 0 )
        {
          v14 = (__int64 *)v9;
          if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
            v14 = **(__int64 ***)(v9 + 16);
          v25 = i;
          v28 = v9;
          v15 = sub_34D06B0(a1, v14);
          v9 = v28;
          i = v25;
          v16 = __OFADD__(v15, v11);
          v11 += v15;
          if ( v16 )
          {
            v11 = 0x8000000000000000LL;
            if ( v15 )
              v11 = 0x7FFFFFFFFFFFFFFFLL;
          }
        }
      }
    }
    if ( *(_BYTE *)(v8 + 8) == 18 )
      goto LABEL_22;
  }
  v17 = *(_DWORD *)(v8 + 32);
  if ( v17 <= 0 )
    goto LABEL_22;
  v18 = 0;
  for ( j = 0; j != v17; ++j )
  {
    v20 = *(_QWORD *)a5;
    if ( *(_DWORD *)(a5 + 8) > 0x40u )
      v20 = *(_QWORD *)(v20 + 8LL * (j >> 6));
    if ( (v20 & (1LL << j)) != 0 )
    {
      v21 = (__int64 *)v8;
      if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
        v21 = **(__int64 ***)(v8 + 16);
      v26 = v18;
      v29 = j;
      v22 = sub_34D06B0(a1, v21);
      j = v29;
      v18 = v22 + v26;
      if ( __OFADD__(v22, v26) )
      {
        v18 = 0x8000000000000000LL;
        if ( v22 )
          v18 = 0x7FFFFFFFFFFFFFFFLL;
      }
    }
  }
  v16 = __OFADD__(v18, v11);
  v11 += v18;
  if ( !v16 )
    goto LABEL_22;
  v23 = 0x7FFFFFFFFFFFFFFFLL;
  if ( v18 <= 0 )
    v23 = 0x8000000000000000LL;
LABEL_23:
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  return v23;
}
