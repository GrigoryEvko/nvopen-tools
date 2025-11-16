// Function: sub_1E63390
// Address: 0x1e63390
//
__int64 __fastcall sub_1E63390(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rcx
  unsigned int v4; // r8d

  v3 = *(_QWORD **)(a2 + 88);
  v4 = 0;
  if ( (unsigned int)((__int64)(*(_QWORD *)(a2 + 96) - (_QWORD)v3) >> 3) <= 1 )
    LOBYTE(v4) = *v3 == a3;
  return v4;
}
