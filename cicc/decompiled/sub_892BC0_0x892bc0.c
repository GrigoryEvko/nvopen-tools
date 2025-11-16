// Function: sub_892BC0
// Address: 0x892bc0
//
__int64 __fastcall sub_892BC0(__int64 a1)
{
  char v1; // dl
  __int64 v2; // rax

  v1 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 80LL);
  v2 = *(_QWORD *)(a1 + 64);
  if ( v1 == 3 )
    return *(_QWORD *)(v2 + 168) + 24LL;
  if ( v1 == 2 )
    return v2 + 184;
  return *(_QWORD *)(v2 + 104) + 128LL;
}
