// Function: sub_34D68F0
// Address: 0x34d68f0
//
unsigned __int64 __fastcall sub_34D68F0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, unsigned int a5)
{
  int v5; // r11d
  unsigned __int64 result; // rax
  int v9; // edx
  unsigned int v10; // esi
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rbx
  unsigned int i; // ecx
  unsigned __int64 v14; // rax
  __int64 *v15; // rsi
  unsigned int v16; // eax
  bool v17; // of
  signed __int64 v18; // rax
  int v19; // edx
  __int64 v20; // rcx
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rbx
  int v23; // [rsp+Ch] [rbp-54h]
  int v24; // [rsp+10h] [rbp-50h]
  unsigned int v25; // [rsp+14h] [rbp-4Ch]
  int v26; // [rsp+14h] [rbp-4Ch]
  unsigned __int64 v28; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-38h]

  v5 = a5;
  if ( !BYTE4(a4) || (a4 & 1) != 0 )
    return sub_34D61B0(a1, a2, (_QWORD **)a3, a5);
  if ( *(_BYTE *)(a3 + 8) == 18 )
    return 0;
  v9 = *(_DWORD *)(a3 + 32);
  v29 = v9;
  if ( (unsigned int)v9 > 0x40 )
  {
    sub_C43690((__int64)&v28, -1, 1);
    v5 = a5;
    if ( *(_BYTE *)(a3 + 8) == 18 )
    {
      v10 = v29;
      v12 = 0;
      goto LABEL_20;
    }
    v9 = *(_DWORD *)(a3 + 32);
    v10 = v29;
  }
  else
  {
    v10 = v9;
    v11 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
    if ( !v9 )
      v11 = 0;
    v28 = v11;
  }
  v12 = 0;
  if ( v9 > 0 )
  {
    for ( i = 0; i != v9; ++i )
    {
      v14 = v28;
      if ( v10 > 0x40 )
        v14 = *(_QWORD *)(v28 + 8LL * (i >> 6));
      if ( (v14 & (1LL << i)) != 0 )
      {
        v15 = (__int64 *)a3;
        if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
          v15 = **(__int64 ***)(a3 + 16);
        v23 = v5;
        v24 = v9;
        v25 = i;
        v16 = sub_34D06B0(a1, v15);
        v10 = v29;
        i = v25;
        v9 = v24;
        v5 = v23;
        v17 = __OFADD__(v16, v12);
        v12 += v16;
        if ( v17 )
        {
          v12 = 0x8000000000000000LL;
          if ( v16 )
            v12 = 0x7FFFFFFFFFFFFFFFLL;
        }
      }
    }
  }
LABEL_20:
  if ( v10 > 0x40 && v28 )
  {
    v26 = v5;
    j_j___libc_free_0_0(v28);
    v5 = v26;
  }
  v18 = sub_34D2250(a1, a2, *(_QWORD *)(a3 + 24), v5, 0, 0, 0, 0, 0);
  v20 = *(unsigned int *)(a3 + 32) * v18;
  if ( !is_mul_ok(*(unsigned int *)(a3 + 32), v18) )
  {
    if ( *(_DWORD *)(a3 + 32) && v18 > 0 )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v19 != 1 )
      {
        v17 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v12);
        v21 = v12 + 0x7FFFFFFFFFFFFFFFLL;
        if ( v17 )
          return result;
        return v21;
      }
      v17 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v12);
      v22 = v12 + 0x7FFFFFFFFFFFFFFFLL;
      if ( v17 )
        return result;
    }
    else
    {
      if ( v19 != 1 )
      {
        v17 = __OFADD__(0x8000000000000000LL, v12);
        v21 = v12 + 0x8000000000000000LL;
        if ( v17 )
          return 0x8000000000000000LL;
        return v21;
      }
      v17 = __OFADD__(0x8000000000000000LL, v12);
      v22 = v12 + 0x8000000000000000LL;
      if ( v17 )
        return 0x8000000000000000LL;
    }
    return v22;
  }
  v17 = __OFADD__(v20, v12);
  v21 = v20 + v12;
  if ( !v17 )
    return v21;
  result = 0x7FFFFFFFFFFFFFFFLL;
  if ( v20 <= 0 )
    return 0x8000000000000000LL;
  return result;
}
