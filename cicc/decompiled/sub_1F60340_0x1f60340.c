// Function: sub_1F60340
// Address: 0x1f60340
//
__int64 __fastcall sub_1F60340(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  char v3; // dl
  _QWORD *v4; // rax
  bool v5; // zf
  __int64 result; // rax
  __int64 v7; // rdx

  v2 = sub_157EBA0(a1);
  v3 = *(_BYTE *)(v2 + 16);
  if ( v3 == 29 )
    return 0;
  if ( v3 == 34 )
  {
    if ( (*(_BYTE *)(v2 + 23) & 0x40) != 0 )
      v4 = *(_QWORD **)(v2 - 8);
    else
      v4 = (_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
    v5 = a2 == *v4;
    result = 0;
    if ( v5 )
      return a1;
  }
  else
  {
    v7 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
    result = 0;
    if ( a2 == *(_QWORD *)(v7 - 24) )
      return *(_QWORD *)(v7 + 40);
  }
  return result;
}
