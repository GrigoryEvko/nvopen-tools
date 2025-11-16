// Function: sub_AC3250
// Address: 0xac3250
//
__int64 __fastcall sub_AC3250(__int64 a1)
{
  __int64 v1; // rdx
  int v2; // eax

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(unsigned __int8 *)(v1 + 8);
  if ( (_BYTE)v2 == 16 || (unsigned int)(v2 - 17) <= 1 )
    return *(unsigned int *)(v1 + 32);
  else
    return *(unsigned int *)(v1 + 12);
}
