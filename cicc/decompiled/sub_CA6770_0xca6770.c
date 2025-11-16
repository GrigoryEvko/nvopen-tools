// Function: sub_CA6770
// Address: 0xca6770
//
__int64 __fastcall sub_CA6770(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int64 v9; // r12

  v7 = a3[1];
  v8 = *a2;
  v9 = a2[1];
  if ( (unsigned __int64)(v7 + 1) > a3[2] )
  {
    sub_C8D290((__int64)a3, a3 + 3, v7 + 1, 1u, a5, a6);
    v7 = a3[1];
  }
  *(_BYTE *)(*a3 + v7) = 39;
  ++a3[1];
  if ( v9 > 1 )
    v9 = 2;
  return v8 + v9;
}
