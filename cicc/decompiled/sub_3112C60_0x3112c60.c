// Function: sub_3112C60
// Address: 0x3112c60
//
void __fastcall sub_3112C60(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  *a1 = &unk_4A32B10;
  v2 = a1[8];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[7];
  *a1 = &unk_4A32AC8;
  if ( v3 )
    sub_31128F0(v3);
  v4 = a1[6];
  if ( v4 )
  {
    sub_3112140(v4 + 16);
    v5 = *(_QWORD *)(v4 + 16);
    if ( v5 != v4 + 64 )
      j_j___libc_free_0(v5);
    j_j___libc_free_0(v4);
  }
  v6 = a1[2];
  if ( (_QWORD *)v6 != a1 + 4 )
    j_j___libc_free_0(v6);
}
