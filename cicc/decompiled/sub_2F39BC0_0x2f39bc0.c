// Function: sub_2F39BC0
// Address: 0x2f39bc0
//
__int64 __fastcall sub_2F39BC0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  *a1 = off_4A2A8D8;
  v2 = a1[445];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[446];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  v4 = (_QWORD *)a1[452];
  v5 = (_QWORD *)a1[451];
  if ( v4 != v5 )
  {
    do
    {
      if ( *v5 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v5 + 16LL))(*v5);
      ++v5;
    }
    while ( v4 != v5 );
    v5 = (_QWORD *)a1[451];
  }
  if ( v5 )
    j_j___libc_free_0((unsigned __int64)v5);
  v6 = a1[448];
  if ( v6 )
    j_j___libc_free_0(v6);
  v7 = a1[442];
  if ( v7 )
    j_j___libc_free_0(v7);
  v8 = a1[438];
  a1[432] = &unk_4A38790;
  if ( v8 )
    j_j___libc_free_0(v8);
  v9 = a1[435];
  if ( v9 )
    j_j___libc_free_0(v9);
  return sub_2EC45A0((__int64)a1);
}
