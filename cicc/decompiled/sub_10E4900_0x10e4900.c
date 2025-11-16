// Function: sub_10E4900
// Address: 0x10e4900
//
__int64 __fastcall sub_10E4900(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( v2 )
  {
    if ( !*(_QWORD *)(v2 + 8) && *(_BYTE *)a2 == 75 )
    {
      v5 = *(_QWORD *)(a2 - 32);
      if ( v5 )
      {
        v3 = 1;
        **a1 = v5;
      }
    }
  }
  return v3;
}
