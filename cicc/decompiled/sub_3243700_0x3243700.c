// Function: sub_3243700
// Address: 0x3243700
//
__int64 __fastcall sub_3243700(_BYTE *a1)
{
  int v1; // eax
  __int64 result; // rax

  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)a1 + 56LL))(a1);
  v1 = (unsigned __int8)a1[100];
  a1[8] = 0;
  result = ((unsigned __int8)v1 >> 3) & 7 | v1 & 0xFFFFFFF8;
  a1[100] = result;
  return result;
}
