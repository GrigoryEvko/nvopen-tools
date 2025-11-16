// Function: sub_14D9090
// Address: 0x14d9090
//
__int64 __fastcall sub_14D9090(__int64 a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v3; // r12
  _QWORD *v4; // rbx

  v3 = &a2[a3];
  if ( a2 != v3 )
  {
    v4 = a2;
    do
    {
      a1 = sub_15A0F90(a1, *v4);
      if ( !a1 )
        break;
      ++v4;
    }
    while ( v3 != v4 );
  }
  return a1;
}
