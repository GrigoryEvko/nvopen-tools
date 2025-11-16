// Function: sub_2EC2400
// Address: 0x2ec2400
//
void __fastcall sub_2EC2400(unsigned __int64 a1)
{
  void (__fastcall *v2)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  *(_QWORD *)a1 = &unk_4A29D28;
  v2 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(a1 + 672);
  if ( v2 )
    v2(a1 + 656, a1 + 656, 3);
  v3 = *(_QWORD *)(a1 + 192);
  qword_5021050[2] = 0;
  if ( v3 != a1 + 208 )
    _libc_free(v3);
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104));
  v4 = *(_QWORD *)(a1 + 72);
  if ( v4 != a1 + 88 )
    _libc_free(v4);
  j_j___libc_free_0(a1);
}
