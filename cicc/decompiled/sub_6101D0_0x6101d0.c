// Function: sub_6101D0
// Address: 0x6101d0
//
__int64 __fastcall sub_6101D0(__int64 a1, const char *a2, char a3, char a4, char a5, int a6, unsigned int a7)
{
  __int64 v8; // rax
  char *v12; // rbx
  size_t v13; // rax

  v8 = dword_4CF8188++;
  if ( (_DWORD)v8 == 590 )
    sub_721090(a1);
  v12 = (char *)&unk_4CF81A0 + 40 * v8;
  *(_DWORD *)v12 = a1;
  *((_QWORD *)v12 + 1) = a2;
  v13 = strlen(a2);
  v12[16] = a3;
  *((_QWORD *)v12 + 3) = v13;
  v12[17] = a4;
  v12[18] = a5;
  *((_DWORD *)v12 + 8) = a6;
  v12[19] = a7;
  return a7;
}
