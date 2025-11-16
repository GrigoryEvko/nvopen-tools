// Function: sub_1D03E50
// Address: 0x1d03e50
//
__int64 __fastcall sub_1D03E50(_QWORD *a1, const char *a2, const char *a3, __int64 a4)
{
  size_t v4; // rax
  size_t v7; // rax

  v4 = 0;
  *a1 = 0;
  a1[1] = a2;
  if ( a2 )
    v4 = strlen(a2);
  a1[2] = v4;
  v7 = 0;
  a1[3] = a3;
  if ( a3 )
    v7 = strlen(a3);
  a1[5] = a4;
  a1[4] = v7;
  return sub_1E40390(qword_4FC1B10, a1);
}
