// Function: sub_7E3E60
// Address: 0x7e3e60
//
_BOOL8 __fastcall sub_7E3E60(__int64 a1, _DWORD *a2)
{
  _BOOL8 result; // rax
  int v3; // [rsp+0h] [rbp-20h] BYREF
  int v4; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v5[3]; // [rsp+8h] [rbp-18h] BYREF

  *a2 = 1;
  if ( (unsigned int)sub_8D23B0(a1) )
    return 0;
  *a2 = 0;
  if ( (*(_BYTE *)(a1 + 176) & 0x50) == 0 )
    return 0;
  result = 1;
  if ( unk_4D03F88 )
  {
    sub_7E3BF0(a1, &v3, v5, &v4);
    return v4 == 0;
  }
  return result;
}
