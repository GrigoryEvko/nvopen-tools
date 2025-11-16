// Function: sub_306A0F0
// Address: 0x306a0f0
//
unsigned __int64 __fastcall sub_306A0F0(__int64 a1, int a2, __int64 a3, __int64 a4, unsigned int a5)
{
  int v5; // r11d
  unsigned int v6; // r10d
  __int64 v7; // r15
  unsigned __int64 result; // rax
  unsigned int v10; // eax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned int v13; // r10d
  unsigned __int64 v14; // r13
  int v15; // r11d
  signed __int64 v16; // rax
  int v17; // edx
  __int64 v18; // rcx
  bool v19; // of
  unsigned __int64 v20; // r13
  int v21; // [rsp+0h] [rbp-50h]
  unsigned int v22; // [rsp+0h] [rbp-50h]
  unsigned int v24; // [rsp+4h] [rbp-4Ch]
  int v25; // [rsp+4h] [rbp-4Ch]
  unsigned __int64 v26; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-38h]

  v5 = a2;
  v6 = a5;
  v7 = a1 + 8;
  if ( !BYTE4(a4) || (a4 & 1) != 0 )
    return sub_3069790(a1 + 8, a2, (_QWORD **)a3, a5);
  if ( *(_BYTE *)(a3 + 8) == 18 )
    return 0;
  v10 = *(_DWORD *)(a3 + 32);
  v27 = v10;
  if ( v10 > 0x40 )
  {
    sub_C43690((__int64)&v26, -1, 1);
    v6 = a5;
    v5 = a2;
  }
  else
  {
    v11 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v10;
    if ( !v10 )
      v11 = 0;
    v26 = v11;
  }
  v21 = v5;
  v24 = v6;
  v12 = sub_3064F80(v7, a3, (__int64 *)&v26, 0, 1);
  v13 = v24;
  v14 = v12;
  v15 = v21;
  if ( v27 > 0x40 && v26 )
  {
    v22 = v24;
    v25 = v15;
    j_j___libc_free_0_0(v26);
    v13 = v22;
    v15 = v25;
  }
  v16 = sub_3075ED0(v7, v15, *(_QWORD *)(a3 + 24), v13, 0, 0, 0, 0, 0);
  v18 = *(unsigned int *)(a3 + 32) * v16;
  if ( !is_mul_ok(*(unsigned int *)(a3 + 32), v16) )
  {
    if ( !*(_DWORD *)(a3 + 32) || v16 <= 0 )
    {
      if ( v17 == 1 )
      {
        result = v14 + 0x8000000000000000LL;
        if ( __OFADD__(v14, 0x8000000000000000LL) )
          return 0x8000000000000000LL;
      }
      else
      {
        result = v14 + 0x8000000000000000LL;
        if ( __OFADD__(v14, 0x8000000000000000LL) )
          return 0x8000000000000000LL;
      }
      return result;
    }
    if ( v17 == 1 )
    {
      result = v14 + 0x7FFFFFFFFFFFFFFFLL;
      if ( __OFADD__(0x7FFFFFFFFFFFFFFFLL, v14) )
        return 0x7FFFFFFFFFFFFFFFLL;
      return result;
    }
    result = 0x7FFFFFFFFFFFFFFFLL;
    v19 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v14);
    v20 = v14 + 0x7FFFFFFFFFFFFFFFLL;
    if ( v19 )
      return result;
    return v20;
  }
  v19 = __OFADD__(v18, v14);
  v20 = v18 + v14;
  if ( !v19 )
    return v20;
  result = 0x7FFFFFFFFFFFFFFFLL;
  if ( v18 <= 0 )
    return 0x8000000000000000LL;
  return result;
}
