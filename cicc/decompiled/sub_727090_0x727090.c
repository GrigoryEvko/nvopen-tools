// Function: sub_727090
// Address: 0x727090
//
_BYTE *sub_727090()
{
  __int64 v0; // rdx
  __int64 v1; // rdx
  _BYTE *result; // rax

  v0 = 0;
  if ( dword_4F07270[0] != unk_4F073B8 )
    v0 = 776LL * dword_4F04C58;
  v1 = qword_4F04C68[0] + v0;
  result = *(_BYTE **)(v1 + 320);
  if ( result )
    *(_QWORD *)(v1 + 320) = *(_QWORD *)result;
  else
    result = sub_7246D0(32);
  *(_QWORD *)result = 0;
  *((_QWORD *)result + 1) = 0;
  result[16] = 0;
  *((_QWORD *)result + 3) = 0;
  return result;
}
