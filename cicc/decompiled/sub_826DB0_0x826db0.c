// Function: sub_826DB0
// Address: 0x826db0
//
__int64 __fastcall sub_826DB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  int v5; // r13d

  v2 = *(_QWORD *)(a2 + 32);
  v3 = *(_QWORD *)(a1 + 32);
  if ( !v3
    || !v2
    || !(unsigned int)sub_8D2FB0(*(_QWORD *)(a1 + 32))
    || !(unsigned int)sub_8D2FB0(v2)
    || *(_BYTE *)(a1 + 15) && !*(_BYTE *)(a1 + 20) )
  {
    return 0;
  }
  if ( *(_BYTE *)(a2 + 15) && !*(_BYTE *)(a2 + 20) )
    return 0;
  v5 = sub_8D3110(v3);
  if ( v5 == (unsigned int)sub_8D3110(v2) )
    return 0;
  if ( v5 )
    return *(_BYTE *)(a1 + 19) == 0 ? 1 : -1;
  return *(_BYTE *)(a2 + 19) == 0 ? -1 : 1;
}
