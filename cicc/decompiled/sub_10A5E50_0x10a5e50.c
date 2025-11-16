// Function: sub_10A5E50
// Address: 0x10a5e50
//
__int64 __fastcall sub_10A5E50(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( v2 )
  {
    if ( !*(_QWORD *)(v2 + 8) && *(_BYTE *)a2 == 46 )
    {
      v5 = *(_QWORD *)(a2 - 64);
      if ( v5 )
      {
        **a1 = v5;
        LOBYTE(v3) = *a1[1] == *(_QWORD *)(a2 - 32);
      }
    }
  }
  return v3;
}
