// Function: sub_1210CA0
// Address: 0x1210ca0
//
__int64 __fastcall sub_1210CA0(__int64 a1)
{
  __int64 result; // rax
  _BYTE *v2; // rdi
  __int64 v3; // [rsp+18h] [rbp-18h] BYREF

  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  result = sub_120AFE0(a1, 16, "expected ':' here");
  if ( !(_BYTE)result )
  {
    result = sub_120C050(a1, &v3);
    if ( !(_BYTE)result )
    {
      v2 = *(_BYTE **)(a1 + 352);
      if ( v2 )
      {
        sub_BAEE70(v2, v3);
        return 0;
      }
    }
  }
  return result;
}
