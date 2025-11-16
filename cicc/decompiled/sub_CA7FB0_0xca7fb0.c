// Function: sub_CA7FB0
// Address: 0xca7fb0
//
__int64 __fastcall sub_CA7FB0(__int64 a1, _BYTE *a2)
{
  unsigned int v2; // r13d
  _QWORD v4[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( *(_BYTE **)(a1 + 48) == a2 )
    return 0;
  v2 = sub_CA7F80(a1, a2);
  if ( (_BYTE)v2 )
  {
    return 0;
  }
  else
  {
    if ( !*(_DWORD *)(a1 + 68) )
      return 1;
    v4[0] = a2;
    v4[1] = 1;
    if ( sub_C934D0(v4, ",[]{}", 5, 0) == -1 )
      return 1;
  }
  return v2;
}
