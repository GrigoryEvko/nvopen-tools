// Function: sub_2A61050
// Address: 0x2a61050
//
__int64 __fastcall sub_2A61050(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 j; // r15
  unsigned int i; // [rsp+14h] [rbp-3Ch]

  v3 = *(_QWORD *)(a2 + 144);
  for ( i = *(_DWORD *)(a2 + 112); a2 + 128 != v3; v3 = sub_220EF30(v3) )
  {
    for ( j = *(_QWORD *)(v3 + 64); v3 + 48 != j; j = sub_220EF30(j) )
    {
      if ( sub_2A60EC0(j + 48, a3, *(_BYTE *)(a1 + 40)) )
        i += sub_2A61050(a1, j + 48, a3);
    }
  }
  return i;
}
