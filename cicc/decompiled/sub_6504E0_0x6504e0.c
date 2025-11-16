// Function: sub_6504E0
// Address: 0x6504e0
//
unsigned int *__fastcall sub_6504E0(__int64 a1, unsigned int a2, __int64 *a3, int a4, int a5, _QWORD *a6)
{
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r15
  char v11; // al
  unsigned int *result; // rax

  v8 = sub_726DD0();
  v9 = *a3;
  *(_QWORD *)(v8 + 24) = a1;
  *(_BYTE *)(v8 + 16) = 28;
  *(_QWORD *)(v8 + 8) = v9;
  *(_BYTE *)(v8 + 40) = *(_BYTE *)(v8 + 40) & 0xCE | (32 * (a5 & 1)) | (16 * (a4 & 1)) | 1;
  v10 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  if ( a5 )
    sub_650490(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C, v8);
  v11 = *(_BYTE *)(v10 + 4);
  if ( (unsigned __int8)(v11 - 3) > 1u && v11 )
    *(_QWORD *)(v8 + 56) = 1;
  else
    *(_QWORD *)(v8 + 56) = ++dword_4F066AC;
  sub_5CEC90(a6, v8, 29);
  sub_7332A0(v8, a2);
  sub_85F7E0(v8, a2);
  result = &dword_4F04C3C;
  if ( !(dword_4F04C3C | a4) )
    return (unsigned int *)sub_8699D0(v8, 29, 0);
  return result;
}
