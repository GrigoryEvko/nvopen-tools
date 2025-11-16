// Function: sub_AA5B10
// Address: 0xaa5b10
//
bool __fastcall sub_AA5B10(__int64 a1)
{
  unsigned __int64 v1; // rax
  int v2; // eax

  v1 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == a1 + 48 )
    return 1;
  if ( !v1 )
    BUG();
  v2 = *(unsigned __int8 *)(v1 - 24);
  if ( (unsigned int)(v2 - 30) > 0xA )
    return 1;
  if ( (unsigned int)(v2 - 29) > 6 )
    return (unsigned int)(v2 - 37) > 3;
  return (unsigned int)(v2 - 29) <= 4;
}
