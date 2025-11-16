// Function: sub_831280
// Address: 0x831280
//
_BOOL8 __fastcall sub_831280(__int64 a1, _QWORD *a2)
{
  _BOOL8 result; // rax
  __int64 v3; // rdx
  char v4; // al
  __int64 v5; // r12
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v6[0] = 0;
  if ( a2 )
    *a2 = 0;
  if ( *(_WORD *)(a1 + 16) != 513 )
    return 0;
  v3 = *(_QWORD *)(a1 + 144);
  v4 = *(_BYTE *)(v3 + 24);
  if ( v4 != 3 )
  {
    if ( v4 == 24 )
      return *(_DWORD *)(v3 + 56) == 0;
    return 0;
  }
  v5 = *(_QWORD *)(v3 + 56);
  if ( !(unsigned int)sub_830310(v6, 0, 0, 0) || v6[0] != v5 || !v6[0] )
    return 0;
  result = 1;
  if ( a2 )
    *a2 = v6[0];
  return result;
}
