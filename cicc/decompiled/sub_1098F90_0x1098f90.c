// Function: sub_1098F90
// Address: 0x1098f90
//
__int64 __fastcall sub_1098F90(unsigned int *a1, char *a2, __int64 a3)
{
  unsigned int v4; // edi
  __int64 result; // rax

  v4 = ~*a1;
  *a1 = v4;
  result = (unsigned int)~sub_1098F50(v4, a2, a3);
  *a1 = result;
  return result;
}
