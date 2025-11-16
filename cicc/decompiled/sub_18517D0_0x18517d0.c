// Function: sub_18517D0
// Address: 0x18517d0
//
__int64 __fastcall sub_18517D0(_QWORD *a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  __int64 *v7; // r9
  unsigned __int64 v8; // r15
  __int64 *v9; // rax
  _QWORD *v10; // rsi
  __int64 result; // rax
  _QWORD *v12; // rax

  v5 = *a2;
  v6 = a1[1];
  v7 = *(__int64 **)(*a1 + 8 * (*a2 % v6));
  v8 = *a2 % v6;
  if ( v7 )
  {
    v9 = (__int64 *)*v7;
    if ( v5 == *(_QWORD *)(*v7 + 8) )
    {
LABEL_6:
      result = *v7;
      if ( *v7 )
        return result;
    }
    else
    {
      while ( 1 )
      {
        v10 = (_QWORD *)*v9;
        if ( !*v9 )
          break;
        v7 = v9;
        if ( v8 != v10[1] % v6 )
          break;
        v9 = (__int64 *)*v9;
        if ( v5 == v10[1] )
          goto LABEL_6;
      }
    }
  }
  v12 = (_QWORD *)sub_22077B0(16);
  if ( v12 )
    *v12 = 0;
  v12[1] = *a2;
  return sub_1851560(a1, v8, v5, (__int64)v12, a3);
}
