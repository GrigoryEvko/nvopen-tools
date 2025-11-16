// Function: sub_34D6B80
// Address: 0x34d6b80
//
unsigned __int64 __fastcall sub_34D6B80(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned int v5; // r10d
  int v6; // r11d
  __int64 v7; // r14
  __int64 v8; // r12
  unsigned __int64 result; // rax
  int v10; // edx
  unsigned int v11; // esi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // r9
  unsigned int v15; // r12d
  unsigned __int64 v16; // r14
  int v17; // r13d
  unsigned __int64 v18; // rax
  __int64 *v19; // rsi
  unsigned int v20; // eax
  bool v21; // of
  unsigned __int64 v22; // rax
  signed __int64 v23; // rax
  int v24; // edx
  __int64 v25; // rcx
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rbx
  int v28; // [rsp+8h] [rbp-58h]
  unsigned int v29; // [rsp+Ch] [rbp-54h]
  int v30; // [rsp+Ch] [rbp-54h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  unsigned int v33; // [rsp+10h] [rbp-50h]
  unsigned __int64 v34; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v35; // [rsp+28h] [rbp-38h]

  v5 = a2;
  v6 = a5;
  v7 = a1 + 8;
  v8 = a3;
  if ( !BYTE4(a4) || (a4 & 1) != 0 )
    return sub_34D61B0(a1 + 8, a2, (_QWORD **)a3, a5);
  if ( *(_BYTE *)(a3 + 8) == 18 )
    return 0;
  v10 = *(_DWORD *)(a3 + 32);
  v35 = v10;
  if ( (unsigned int)v10 > 0x40 )
  {
    sub_C43690((__int64)&v34, -1, 1);
    v5 = a2;
    v6 = a5;
    if ( *(_BYTE *)(v8 + 8) == 18 )
    {
      v11 = v35;
      v13 = 0;
      goto LABEL_21;
    }
    v10 = *(_DWORD *)(v8 + 32);
    v11 = v35;
  }
  else
  {
    v11 = v10;
    v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v10;
    if ( !v10 )
      v12 = 0;
    v34 = v12;
  }
  v13 = 0;
  if ( v10 > 0 )
  {
    v29 = v5;
    v28 = v6;
    v14 = v8;
    v15 = 0;
    v16 = 0;
    v17 = v10;
    do
    {
      v18 = v34;
      if ( v11 > 0x40 )
        v18 = *(_QWORD *)(v34 + 8LL * (v15 >> 6));
      if ( (v18 & (1LL << v15)) != 0 )
      {
        v19 = (__int64 *)v14;
        if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
          v19 = **(__int64 ***)(v14 + 16);
        v32 = v14;
        v20 = sub_34D06B0(a1 + 8, v19);
        v11 = v35;
        v14 = v32;
        v21 = __OFADD__(v20, v16);
        v16 += v20;
        if ( v21 )
        {
          v16 = 0x8000000000000000LL;
          if ( v20 )
            v16 = 0x7FFFFFFFFFFFFFFFLL;
        }
      }
      ++v15;
    }
    while ( v17 != v15 );
    v22 = v16;
    v5 = v29;
    v6 = v28;
    v7 = a1 + 8;
    v8 = v14;
    v13 = v22;
  }
LABEL_21:
  if ( v11 > 0x40 && v34 )
  {
    v30 = v6;
    v33 = v5;
    j_j___libc_free_0_0(v34);
    v6 = v30;
    v5 = v33;
  }
  v23 = sub_34D2250(v7, v5, *(_QWORD *)(v8 + 24), v6, 0, 0, 0, 0, 0);
  v25 = *(unsigned int *)(v8 + 32) * v23;
  if ( !is_mul_ok(*(unsigned int *)(v8 + 32), v23) )
  {
    if ( *(_DWORD *)(v8 + 32) && v23 > 0 )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v24 != 1 )
      {
        v21 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v13);
        v26 = v13 + 0x7FFFFFFFFFFFFFFFLL;
        if ( v21 )
          return result;
        return v26;
      }
      v21 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v13);
      v27 = v13 + 0x7FFFFFFFFFFFFFFFLL;
      if ( v21 )
        return result;
    }
    else
    {
      if ( v24 != 1 )
      {
        v21 = __OFADD__(0x8000000000000000LL, v13);
        v26 = v13 + 0x8000000000000000LL;
        if ( v21 )
          return 0x8000000000000000LL;
        return v26;
      }
      v21 = __OFADD__(0x8000000000000000LL, v13);
      v27 = v13 + 0x8000000000000000LL;
      if ( v21 )
        return 0x8000000000000000LL;
    }
    return v27;
  }
  v21 = __OFADD__(v25, v13);
  v26 = v25 + v13;
  if ( !v21 )
    return v26;
  result = 0x7FFFFFFFFFFFFFFFLL;
  if ( v25 <= 0 )
    return 0x8000000000000000LL;
  return result;
}
