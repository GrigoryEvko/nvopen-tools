// Function: sub_1D18C70
// Address: 0x1d18c70
//
__int64 __fastcall sub_1D18C70(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = *(_QWORD *)(a2 + 48);
  if ( v2 )
  {
    while ( a1 == *(_QWORD *)(v2 + 16) )
    {
      v2 = *(_QWORD *)(v2 + 32);
      if ( !v2 )
        return 1;
    }
  }
  return 0;
}
