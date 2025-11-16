// Function: sub_14EA410
// Address: 0x14ea410
//
__int64 *__fastcall sub_14EA410(__int64 *a1, int a2, __int64 *a3)
{
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rax

  v3 = *a3;
  v4 = a3[1];
  *a3 = 0;
  v5 = a3[2];
  a3[1] = 0;
  a3[2] = 0;
  v6 = sub_22077B0(64);
  if ( v6 )
  {
    *(_DWORD *)(v6 + 8) = 2;
    *(_DWORD *)(v6 + 12) = a2;
    *(_QWORD *)(v6 + 16) = 0;
    *(_QWORD *)(v6 + 24) = 0;
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 40) = v3;
    *(_QWORD *)(v6 + 48) = v4;
    *(_QWORD *)(v6 + 56) = v5;
    *(_QWORD *)v6 = &unk_49EB4D8;
    *a1 = v6;
  }
  else
  {
    *a1 = 0;
    if ( v3 )
      j_j___libc_free_0(v3, v5 - v3);
  }
  return a1;
}
