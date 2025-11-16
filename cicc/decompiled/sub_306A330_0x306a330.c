// Function: sub_306A330
// Address: 0x306a330
//
unsigned __int64 __fastcall sub_306A330(__int64 a1, char a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  _QWORD **v7; // r15
  signed __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  bool v12; // of
  unsigned __int64 v13; // rbx
  unsigned __int64 result; // rax
  bool v15; // cc
  unsigned __int64 v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+28h] [rbp-38h]

  BYTE4(v17) = *(_BYTE *)(a4 + 8) == 18;
  LODWORD(v17) = *(_DWORD *)(a4 + 32);
  v7 = (_QWORD **)sub_BCE1B0(a3, v17);
  v16 = sub_3069790(a1 + 8, 13, v7, a5);
  v8 = sub_3065900(a1 + 8, (unsigned int)(a2 == 0) + 39, (__int64)v7, a4, 0, a5, 0);
  v9 = sub_3075ED0((int)a1 + 8, 17, (_DWORD)v7, a5, 0, 0, 0, 0, 0);
  v10 = 2 * v8;
  if ( !is_mul_ok(2u, v8) )
  {
    v10 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v8 <= 0 )
      v10 = 0x8000000000000000LL;
  }
  v11 = v9 + v16;
  if ( __OFADD__(v9, v16) )
  {
    v11 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v9 <= 0 )
      v11 = 0x8000000000000000LL;
  }
  v12 = __OFADD__(v10, v11);
  v13 = v10 + v11;
  if ( !v12 )
    return v13;
  v15 = v10 <= 0;
  result = 0x7FFFFFFFFFFFFFFFLL;
  if ( v15 )
    return 0x8000000000000000LL;
  return result;
}
