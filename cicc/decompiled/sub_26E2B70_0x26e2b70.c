// Function: sub_26E2B70
// Address: 0x26e2b70
//
void __fastcall sub_26E2B70(unsigned __int64 **a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 *v4; // r13
  _DWORD *v5; // rax
  _QWORD *v6; // rax
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r15
  _DWORD *v10; // rax
  unsigned __int64 v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = *a1;
  v11[0] = a2;
  v5 = sub_26E2B00(v4, a2 % v4[1], v11, a2);
  if ( !v5 || !*(_QWORD *)v5 )
  {
    v6 = (_QWORD *)sub_22077B0(0x20u);
    v7 = (unsigned __int64)v6;
    if ( v6 )
      *v6 = 0;
    v8 = v11[0];
    *(_QWORD *)(v7 + 16) = a3;
    *(_QWORD *)(v7 + 8) = v8;
    v9 = a2 % v4[1];
    v10 = sub_26E2B00(v4, v9, (_DWORD *)(v7 + 8), a2);
    if ( v10 && *(_QWORD *)v10 )
      j_j___libc_free_0(v7);
    else
      sub_26DFEE0(v4, v9, a2, (_QWORD *)v7, 1);
  }
}
