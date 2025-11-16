// Function: sub_22406A0
// Address: 0x22406a0
//
__int64 __fastcall sub_22406A0(_QWORD *a1)
{
  __int64 (*v1)(); // rax
  unsigned int *v3; // rax
  unsigned int v4; // r8d

  v1 = *(__int64 (**)())(*a1 + 72LL);
  if ( v1 == sub_2240420 )
    return 0xFFFFFFFFLL;
  if ( (unsigned int)v1() == -1 )
    return 0xFFFFFFFFLL;
  v3 = (unsigned int *)a1[2];
  v4 = *v3;
  a1[2] = v3 + 1;
  return v4;
}
