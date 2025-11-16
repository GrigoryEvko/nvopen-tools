// Function: sub_880920
// Address: 0x880920
//
__int64 __fastcall sub_880920(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rbx
  char v4; // al

  if ( !(dword_4F077BC | unk_4D043E4) || dword_4F04C64 == -1 )
    return 0;
  v2 = dword_4F04C64;
  do
  {
    v3 = qword_4F04C68[0] + 776 * v2;
    if ( !v3 )
      break;
    v4 = *(_BYTE *)(v3 + 4);
    if ( ((unsigned __int8)(v4 - 3) <= 1u || !v4) && (unsigned int)sub_880800(a1, *(_QWORD *)(v3 + 184)) )
      return 1;
    v2 = *(int *)(v3 + 552);
  }
  while ( (_DWORD)v2 != -1 );
  return 0;
}
