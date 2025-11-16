// Function: sub_3075ED0
// Address: 0x3075ed0
//
__int64 __fastcall sub_3075ED0(__int64 a1, unsigned int a2, __int64 *a3, int a4, int a5, int a6)
{
  __int64 v6; // r15
  __int64 v7; // rcx
  __int16 v8; // bx
  __int64 v9; // rdx
  __int64 v10; // r13
  __int16 v11; // dx
  __int64 v12; // r12
  int v13; // eax
  __int64 result; // rax
  __int64 v15; // rax
  __int16 v19; // [rsp+18h] [rbp-68h]
  signed __int64 v21; // [rsp+28h] [rbp-58h]
  _BYTE v22[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v23; // [rsp+38h] [rbp-48h]
  __int64 v24; // [rsp+40h] [rbp-40h]

  v6 = *a3;
  v21 = 1;
  v7 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), a3, 0);
  v8 = v7;
  v10 = v9;
  while ( 1 )
  {
    LOWORD(v7) = v8;
    sub_2FE6CC0((__int64)v22, *(_QWORD *)(a1 + 24), v6, v7, v10);
    v11 = v23;
    if ( v22[0] == 10 )
      break;
    if ( !v22[0] )
    {
      v12 = (__int64)a3;
      v11 = v8;
      goto LABEL_8;
    }
    if ( (v22[0] & 0xFB) == 2 )
    {
      v15 = 2 * v21;
      if ( !is_mul_ok(2u, v21) )
      {
        v15 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v21 <= 0 )
          v15 = 0x8000000000000000LL;
      }
      v21 = v15;
    }
    if ( v8 == (_WORD)v23 && ((_WORD)v23 || v10 == v24) )
    {
      v12 = (__int64)a3;
      goto LABEL_8;
    }
    v7 = v23;
    v10 = v24;
    v8 = v23;
  }
  v11 = 8;
  v12 = (__int64)a3;
  v21 = 0;
  if ( v8 )
    v11 = v8;
LABEL_8:
  v19 = v11;
  v13 = sub_2FEBEF0(*(_QWORD *)(a1 + 24), a2);
  if ( v13 != 58 )
  {
    if ( v13 > 58 )
    {
      if ( (unsigned int)(v13 - 186) > 2 || v19 != 8 )
        return sub_3075480(a1, a2, v12, a4, a5, a6, 0, 0, 0);
      goto LABEL_17;
    }
    if ( v13 != 56 )
      return sub_3075480(a1, a2, v12, a4, a5, a6, 0, 0, 0);
  }
  if ( v19 != 8 )
    return sub_3075480(a1, a2, v12, a4, a5, a6, 0, 0, 0);
LABEL_17:
  result = 2 * v21;
  if ( !is_mul_ok(2u, v21) )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v21 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
