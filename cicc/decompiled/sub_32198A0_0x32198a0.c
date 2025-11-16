// Function: sub_32198A0
// Address: 0x32198a0
//
void __fastcall sub_32198A0(unsigned __int64 a1)
{
  __int64 v1; // rax

  *(_QWORD *)a1 = &unk_4A35790;
  v1 = *(char *)(a1 + 88);
  if ( (_BYTE)v1 != 0xFF )
    funcs_32198D3[v1]();
  j_j___libc_free_0(a1);
}
