// Function: sub_3157390
// Address: 0x3157390
//
__int64 __fastcall sub_3157390(_QWORD *a1, _DWORD *a2, __int64 a3)
{
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  char v6; // r14
  char v7; // al

  v4 = a1[116];
  if ( v4 )
  {
    sub_C7D6A0(*(_QWORD *)(v4 + 8), 16LL * *(unsigned int *)(v4 + 24), 8);
    j_j___libc_free_0(v4);
  }
  v5 = sub_22077B0(0x20u);
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_DWORD *)(v5 + 24) = 0;
  }
  a1[116] = v5;
  v6 = *(_DWORD *)(a3 + 636) == 4;
  v7 = sub_23CF1B0(a3);
  sub_E89910((__int64)a1, a2, v7, v6);
  a1[122] = a3;
  *(_QWORD *)((char *)a1 + 940) = 0;
  *(_QWORD *)((char *)a1 + 948) = 0x100000000LL;
  return 0x100000000LL;
}
