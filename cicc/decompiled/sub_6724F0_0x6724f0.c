// Function: sub_6724F0
// Address: 0x6724f0
//
__int64 sub_6724F0()
{
  __int64 v0; // rax
  char v1; // dl
  __int64 v2; // r8

  v0 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v1 = *(_BYTE *)(v0 + 4);
  if ( (unsigned __int8)(v1 - 8) <= 1u )
  {
    v1 = *(_BYTE *)(v0 - 772);
    v0 -= 776;
  }
  if ( (unsigned __int8)(v1 - 6) <= 1u )
    return *(_QWORD *)(v0 + 208);
  v2 = 0;
  if ( v1 == 9 )
    return *(_QWORD *)(v0 + 208);
  return v2;
}
