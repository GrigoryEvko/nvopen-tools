// Function: sub_85FB30
// Address: 0x85fb30
//
__int64 __fastcall sub_85FB30(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax

  v1 = dword_4F04C64;
  do
  {
    v2 = qword_4F04C68[0] + 776 * v1;
    if ( !v2 )
      break;
    if ( (unsigned __int8)(*(_BYTE *)(v2 + 4) - 6) <= 1u && *(_QWORD *)(v2 + 208) == a1 )
      return 1;
    v1 = *(int *)(v2 + 552);
  }
  while ( (_DWORD)v1 != -1 );
  return 0;
}
