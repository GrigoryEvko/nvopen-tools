// Function: sub_10C3FA0
// Address: 0x10c3fa0
//
__int64 __fastcall sub_10C3FA0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( v2 )
  {
    if ( !*(_QWORD *)(v2 + 8) && *(_BYTE *)a2 == 68 )
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
