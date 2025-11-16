// Function: sub_1002420
// Address: 0x1002420
//
bool __fastcall sub_1002420(__int64 *a1)
{
  unsigned int v1; // esi
  __int64 v2; // rax

  v1 = *((_DWORD *)a1 + 2);
  v2 = *a1;
  if ( v1 > 0x40 )
    v2 = *(_QWORD *)(v2 + 8LL * ((v1 - 1) >> 6));
  return (v2 & (1LL << ((unsigned __int8)v1 - 1))) == 0;
}
