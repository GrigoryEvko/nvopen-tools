// Function: sub_986580
// Address: 0x986580
//
unsigned __int64 __fastcall sub_986580(__int64 a1)
{
  unsigned __int64 v1; // rax
  int v2; // edx
  unsigned __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == a1 + 48 )
    return 0;
  if ( !v1 )
    BUG();
  v2 = *(unsigned __int8 *)(v1 - 24);
  result = v1 - 24;
  if ( (unsigned int)(v2 - 30) >= 0xB )
    return 0;
  return result;
}
