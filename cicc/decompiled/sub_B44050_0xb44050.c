// Function: sub_B44050
// Address: 0xb44050
//
__int64 __fastcall sub_B44050(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 *v9; // r12

  result = sub_AA6160(a2, a3);
  v8 = a2 + 48;
  v9 = (__int64 *)result;
  if ( !result || (result = *(_QWORD *)(result + 8) & 0xFFFFFFFFFFFFFFF8LL, (__int64 *)result == v9 + 1) )
  {
LABEL_5:
    if ( a3 != v8 )
      return result;
LABEL_11:
    sub_B14200(v9);
    return sub_AA6260(a2);
  }
  if ( *(_QWORD *)(a1 + 64) )
  {
    sub_AA4580(*(_QWORD *)(a1 + 40), a1);
    result = (__int64)sub_B14410(*(_QWORD *)(a1 + 64), (__int64)v9, a5);
    v8 = a2 + 48;
    goto LABEL_5;
  }
  if ( a3 == v8 )
  {
    sub_AA4580(*(_QWORD *)(a1 + 40), a1);
    sub_B14410(*(_QWORD *)(a1 + 64), (__int64)v9, a5);
    goto LABEL_11;
  }
  *(_QWORD *)(a1 + 64) = v9;
  *v9 = a1;
  if ( !a3 )
  {
    MEMORY[0x40] = 0;
    BUG();
  }
  *(_QWORD *)(a3 + 40) = 0;
  return result;
}
