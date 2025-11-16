// Function: sub_BAA8B0
// Address: 0xbaa8b0
//
__int64 __fastcall sub_BAA8B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // rdx

  v1 = sub_BA91D0(a1, "MaxTLSAlign", 0xBu);
  if ( !v1 )
    return 0;
  v2 = v1;
  result = 0;
  if ( *(_BYTE *)v2 == 1 )
  {
    v4 = *(_QWORD *)(v2 + 136);
    if ( *(_BYTE *)v4 == 17 )
    {
      result = *(_QWORD *)(v4 + 24);
      if ( *(_DWORD *)(v4 + 32) > 0x40u )
        return *(_QWORD *)result;
    }
  }
  return result;
}
