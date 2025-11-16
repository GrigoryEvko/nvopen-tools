// Function: sub_2207F40
// Address: 0x2207f40
//
__off64_t __fastcall sub_2207F40(FILE **a1, __off64_t a2, int a3)
{
  int v4; // eax

  v4 = sub_2207D30(a1);
  return lseek64(v4, a2, a3);
}
