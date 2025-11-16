// Function: sub_256B180
// Address: 0x256b180
//
__int64 __fastcall sub_256B180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdi
  _BYTE v9[48]; // [rsp+0h] [rbp-30h] BYREF

  v6 = *(_QWORD *)(a1 + 64);
  *(_DWORD *)(a1 + 8) = 0;
  while ( v6 )
  {
    sub_253B2D0(*(_QWORD *)(v6 + 24));
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
    j_j___libc_free_0(v7);
  }
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 56;
  *(_QWORD *)(a1 + 80) = a1 + 56;
  *(_QWORD *)(a1 + 88) = 0;
  return sub_256AFA0((__int64)v9, a1, &qword_438A698, a4, a5);
}
