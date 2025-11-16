// Function: sub_352FA10
// Address: 0x352fa10
//
char __fastcall sub_352FA10(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v4; // rax
  char v5; // dl
  unsigned int *v6; // rdx
  unsigned __int64 v7; // r8
  unsigned int v8; // edx
  char result; // al
  char v10; // [rsp+8h] [rbp-18h]

  v4 = sub_2E39F50(a2, a1);
  v10 = v5;
  v6 = *(unsigned int **)(a3 + 8);
  v7 = (unsigned __int64)v4;
  if ( !v6 )
    return (unsigned int)qword_503D668 > v7;
  v8 = *v6;
  result = v10;
  if ( v8 > 1 )
  {
    if ( v8 != 2 || v10 )
      return (unsigned int)qword_503D668 > v7;
  }
  else if ( v10 )
  {
    if ( !(_DWORD)qword_503D748 )
      return (unsigned int)qword_503D668 > v7;
    return sub_D853A0(a3, qword_503D748, v7);
  }
  else
  {
    return 1;
  }
  return result;
}
