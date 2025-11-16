// Function: sub_157EBA0
// Address: 0x157eba0
//
unsigned __int64 __fastcall sub_157EBA0(__int64 a1)
{
  unsigned __int64 v1; // rax
  int v2; // edx
  unsigned __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == a1 + 40 )
    return 0;
  if ( !v1 )
    BUG();
  v2 = *(unsigned __int8 *)(v1 - 8);
  result = v1 - 24;
  if ( (unsigned int)(v2 - 25) >= 0xA )
    return 0;
  return result;
}
