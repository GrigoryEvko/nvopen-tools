// Function: sub_3069EF0
// Address: 0x3069ef0
//
unsigned __int64 __fastcall sub_3069EF0(__int64 a1, int a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int64 result; // rax
  unsigned int v8; // eax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  signed __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rcx
  bool v14; // of
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-38h]

  if ( !BYTE4(a4) || (a4 & 1) != 0 )
    return sub_3069790(a1, a2, (_QWORD **)a3, a5);
  if ( *(_BYTE *)(a3 + 8) == 18 )
    return 0;
  v8 = *(_DWORD *)(a3 + 32);
  v18 = v8;
  if ( v8 > 0x40 )
  {
    sub_C43690((__int64)&v17, -1, 1);
  }
  else
  {
    v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
    if ( !v8 )
      v9 = 0;
    v17 = v9;
  }
  v10 = sub_3064F80(a1, a3, (__int64 *)&v17, 0, 1);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  v11 = sub_3075ED0(a1, a2, *(_QWORD *)(a3 + 24), a5, 0, 0, 0, 0, 0);
  v13 = *(unsigned int *)(a3 + 32) * v11;
  if ( !is_mul_ok(*(unsigned int *)(a3 + 32), v11) )
  {
    if ( *(_DWORD *)(a3 + 32) && v11 > 0 )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v12 != 1 )
      {
        v14 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v10);
        v15 = v10 + 0x7FFFFFFFFFFFFFFFLL;
        if ( v14 )
          return result;
        return v15;
      }
      v14 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v10);
      v16 = v10 + 0x7FFFFFFFFFFFFFFFLL;
      if ( v14 )
        return result;
    }
    else
    {
      if ( v12 != 1 )
      {
        v14 = __OFADD__(0x8000000000000000LL, v10);
        v15 = v10 + 0x8000000000000000LL;
        if ( v14 )
          return 0x8000000000000000LL;
        return v15;
      }
      v14 = __OFADD__(0x8000000000000000LL, v10);
      v16 = v10 + 0x8000000000000000LL;
      if ( v14 )
        return 0x8000000000000000LL;
    }
    return v16;
  }
  v14 = __OFADD__(v13, v10);
  v15 = v13 + v10;
  if ( !v14 )
    return v15;
  result = 0x7FFFFFFFFFFFFFFFLL;
  if ( v13 <= 0 )
    return 0x8000000000000000LL;
  return result;
}
