// Function: sub_5F03A0
// Address: 0x5f03a0
//
__int64 __fastcall sub_5F03A0(__int64 a1, __int64 a2, int a3)
{
  __int64 i; // rax
  __int64 v5; // r13
  __int64 result; // rax
  _BYTE v7[36]; // [rsp+Ch] [rbp-24h] BYREF

  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v5 = *(_QWORD *)(*(_QWORD *)i + 96LL);
  result = sub_72F500(a1, a2, v7, unk_4D04470, 1);
  if ( (_DWORD)result )
  {
    result = unk_4D04470;
    *(_BYTE *)(v5 + 176) |= 8u;
    if ( (_DWORD)result )
    {
      result = sub_72F570(a1);
      if ( (_DWORD)result )
        *(_BYTE *)(v5 + 176) |= 0x40u;
    }
    if ( (v7[0] & 1) != 0 )
      *(_BYTE *)(v5 + 176) |= 0x10u;
    if ( !a3 && (*(_BYTE *)(a1 + 206) & 0x18) == 0 )
      *(_BYTE *)(v5 + 176) |= 0x20u;
  }
  else if ( !a3 )
  {
    result = sub_72F570(a1);
    if ( (_DWORD)result )
    {
      result = *(unsigned __int8 *)(v5 + 176);
      *(_BYTE *)(v5 + 176) |= 0x40u;
      if ( (*(_BYTE *)(a1 + 206) & 0x18) == 0 )
      {
        result = (unsigned int)result | 0xFFFFFFC0;
        *(_BYTE *)(v5 + 176) = result;
      }
    }
  }
  return result;
}
