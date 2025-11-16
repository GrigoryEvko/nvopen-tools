// Function: sub_2D17350
// Address: 0x2d17350
//
__int64 __fastcall sub_2D17350(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  unsigned int v6; // r12d
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rdi
  unsigned __int64 v10[42]; // [rsp+0h] [rbp-150h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  memset(v10, 0, 0x128u);
  v10[7] = (unsigned __int64)&v10[5];
  v10[8] = (unsigned __int64)&v10[5];
  v10[13] = (unsigned __int64)&v10[11];
  v10[14] = (unsigned __int64)&v10[11];
  v10[19] = (unsigned __int64)&v10[17];
  v10[20] = (unsigned __int64)&v10[17];
  v10[25] = (unsigned __int64)&v10[23];
  v10[26] = (unsigned __int64)&v10[23];
  v10[31] = (unsigned __int64)&v10[29];
  v10[32] = (unsigned __int64)&v10[29];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_12;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v6 = sub_2D13E90((__int64)v10, a2, v5 + 176);
  if ( v10[34] )
    j_j___libc_free_0(v10[34]);
  sub_2D0FFC0(v10[30]);
  sub_2D0FDF0(v10[24]);
  sub_2D0FDF0(v10[18]);
  sub_2D0FDF0(v10[12]);
  v7 = v10[6];
  while ( v7 )
  {
    sub_2D0F560(*(_QWORD *)(v7 + 24));
    v8 = v7;
    v7 = *(_QWORD *)(v7 + 16);
    j_j___libc_free_0(v8);
  }
  if ( v10[0] )
    j_j___libc_free_0(v10[0]);
  return v6;
}
