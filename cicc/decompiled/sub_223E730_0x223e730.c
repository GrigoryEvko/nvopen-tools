// Function: sub_223E730
// Address: 0x223e730
//
__int64 *__fastcall sub_223E730(__int64 *a1, unsigned int a2)
{
  int v2; // eax

  v2 = *(_DWORD *)((_BYTE *)a1 + *(_QWORD *)(*a1 - 24) + 24) & 0x4A;
  if ( v2 == 64 || v2 == 8 )
    return sub_223E530(a1, a2);
  else
    return sub_223E530(a1, (int)a2);
}
