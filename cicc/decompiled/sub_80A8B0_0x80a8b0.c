// Function: sub_80A8B0
// Address: 0x80a8b0
//
__int64 __fastcall sub_80A8B0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  char v3; // dl
  __int64 v4; // rdx

  result = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 192LL);
  if ( !result )
    return result;
  while ( 1 )
  {
    v3 = *(_BYTE *)(result + 80);
    if ( v3 == 10 )
    {
      v4 = *(_QWORD *)(result + 88);
      goto LABEL_7;
    }
    if ( v3 == 20 )
      break;
LABEL_4:
    result = *(_QWORD *)(result + 16);
    if ( !result )
      return result;
  }
  v4 = *(_QWORD *)(*(_QWORD *)(result + 88) + 176LL);
LABEL_7:
  if ( !v4 || *(_BYTE *)(v4 + 174) != 5 || *(_BYTE *)(v4 + 176) != 42 )
    goto LABEL_4;
  if ( a2 )
    *a2 = (*(_BYTE *)(v4 + 195) & 0x10) != 0;
  return *(_QWORD *)(v4 + 152);
}
