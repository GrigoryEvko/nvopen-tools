// Function: sub_F4F7F0
// Address: 0xf4f7f0
//
__int64 __fastcall sub_F4F7F0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  _BYTE *v6; // rdi
  __int64 v7; // rax

  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL);
  v5 = *(_BYTE *)(v4 - 16);
  if ( (v5 & 2) != 0 )
  {
    v6 = *(_BYTE **)(*(_QWORD *)(v4 - 32) + 24LL);
    if ( *v6 != 12 )
      return v2;
  }
  else
  {
    v6 = *(_BYTE **)(v4 - 16 - 8LL * ((v5 >> 2) & 0xF) + 24);
    if ( *v6 != 12 )
      return v2;
  }
  v7 = sub_AF2C80((__int64)v6);
  if ( BYTE4(v7) )
    return sub_B0E430(
             *(_QWORD *)(*(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL),
             **a1,
             *a1[1],
             (_DWORD)v7 == 0);
  return v2;
}
