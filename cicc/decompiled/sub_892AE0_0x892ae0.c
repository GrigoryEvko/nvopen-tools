// Function: sub_892AE0
// Address: 0x892ae0
//
__int64 __fastcall sub_892AE0(__int64 a1)
{
  char v1; // dl
  __int64 v2; // rax

  v1 = *(_BYTE *)(a1 + 80);
  v2 = *(_QWORD *)(a1 + 88);
  if ( v1 == 3 )
    return *(_QWORD *)(v2 + 168) + 24LL;
  if ( v1 == 2 )
    return v2 + 184;
  return *(_QWORD *)(v2 + 104) + 128LL;
}
