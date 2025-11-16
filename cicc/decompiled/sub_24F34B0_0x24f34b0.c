// Function: sub_24F34B0
// Address: 0x24f34b0
//
__int64 __fastcall sub_24F34B0(__int64 *a1, _QWORD **a2, __int64 a3)
{
  _QWORD **v3; // rbx
  _QWORD **v4; // r14
  __int64 result; // rax
  __int64 v6; // r13
  _QWORD *v7; // r12

  v3 = a2;
  v4 = &a2[a3];
  result = sub_ACD720(a1);
  if ( a2 != v4 )
  {
    v6 = result;
    do
    {
      v7 = *v3++;
      sub_BD84D0((__int64)v7, v6);
      result = sub_B43D60(v7);
    }
    while ( v4 != v3 );
  }
  return result;
}
