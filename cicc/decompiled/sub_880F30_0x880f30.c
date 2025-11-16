// Function: sub_880F30
// Address: 0x880f30
//
__int64 __fastcall sub_880F30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d

  v2 = *(int *)(a1 + 40);
  v3 = 1;
  if ( (_DWORD)v2 != -1 )
    return *((_QWORD *)qword_4F066A0 + v2) == a2;
  return v3;
}
