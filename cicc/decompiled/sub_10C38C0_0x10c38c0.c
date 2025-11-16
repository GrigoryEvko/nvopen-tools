// Function: sub_10C38C0
// Address: 0x10c38c0
//
__int64 __fastcall sub_10C38C0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax
  __int64 v6; // rax

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( v2 && !*(_QWORD *)(v2 + 8) )
  {
    if ( *(_BYTE *)a2 == 68 )
    {
      v5 = *(_QWORD *)(a2 - 32);
      if ( v5 )
      {
        v3 = 1;
        **a1 = v5;
      }
    }
    else if ( *(_BYTE *)a2 == 69 )
    {
      v6 = *(_QWORD *)(a2 - 32);
      if ( v6 )
      {
        v3 = 1;
        *a1[1] = v6;
      }
    }
  }
  return v3;
}
