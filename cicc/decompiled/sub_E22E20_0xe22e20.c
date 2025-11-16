// Function: sub_E22E20
// Address: 0xe22e20
//
__int64 __fastcall sub_E22E20(__int64 a1, _QWORD *a2)
{
  unsigned __int8 *v2; // rax
  int v3; // edx

  v2 = (unsigned __int8 *)a2[1];
  v3 = *v2;
  --*a2;
  a2[1] = v2 + 1;
  return (unsigned int)(v3 - 47);
}
