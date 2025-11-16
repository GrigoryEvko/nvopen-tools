// Function: sub_2240650
// Address: 0x2240650
//
__int64 __fastcall sub_2240650(_QWORD *a1)
{
  __int64 (*v1)(); // rax
  unsigned __int8 *v3; // rax
  unsigned int v4; // r8d

  v1 = *(__int64 (**)())(*a1 + 72LL);
  if ( v1 == sub_2240390 )
    return 0xFFFFFFFFLL;
  if ( (unsigned int)v1() == -1 )
    return 0xFFFFFFFFLL;
  v3 = (unsigned __int8 *)a1[2];
  v4 = *v3;
  a1[2] = v3 + 1;
  return v4;
}
