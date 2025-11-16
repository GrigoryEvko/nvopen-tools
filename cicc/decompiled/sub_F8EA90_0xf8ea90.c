// Function: sub_F8EA90
// Address: 0xf8ea90
//
bool __fastcall sub_F8EA90(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rbx
  unsigned __int8 *v3; // r12

  v1 = a1 + 48;
  v2 = *(_QWORD *)(a1 + 56);
  if ( a1 + 48 != v2 )
  {
    while ( 1 )
    {
      v3 = (unsigned __int8 *)(v2 - 24);
      if ( !v2 )
        v3 = 0;
      if ( (unsigned __int8)sub_B46490((__int64)v3) || (unsigned __int8)sub_B46970(v3) )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v1 == v2 )
        return 1;
    }
  }
  return v1 == v2;
}
