// Function: sub_8A4460
// Address: 0x8a4460
//
__int64 *__fastcall sub_8A4460(unsigned int *a1, __int16 a2, __int64 *a3, __int64 a4)
{
  int v4; // eax
  unsigned int v7; // r15d

  v4 = 0;
  v7 = a1[1];
  if ( a4 )
    v4 = *(_DWORD *)(sub_892BC0(a4) + 4);
  if ( v7 == v4 )
    return sub_8A4360(a4, a3, a1, 1, (a2 & 0x2000) != 0);
  if ( !qword_4F04C18 || *((_BYTE *)qword_4F04C18 + 42) )
    return 0;
  return sub_866700(a1, 1, 0, 0);
}
