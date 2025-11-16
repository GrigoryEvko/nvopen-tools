// Function: sub_C481A0
// Address: 0xc481a0
//
__int64 __fastcall sub_C481A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r15d
  unsigned int v5; // r14d
  unsigned __int64 v7; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a3 + 8);
  v5 = *(_DWORD *)(a2 + 8);
  if ( v4 > 0x40 )
  {
    v7 = *(unsigned int *)(a2 + 8);
    if ( v4 - (unsigned int)sub_C444A0(a3) <= 0x40 && v7 >= **(_QWORD **)a3 )
      v5 = **(_QWORD **)a3;
  }
  else if ( (unsigned __int64)*(unsigned int *)(a2 + 8) >= *(_QWORD *)a3 )
  {
    v5 = *(_QWORD *)a3;
  }
  sub_C48030(a1, a2, v5);
  return a1;
}
