// Function: sub_BB89D0
// Address: 0xbb89d0
//
void __fastcall sub_BB89D0(_QWORD *a1, char *a2)
{
  void (__fastcall *v3)(char *, char *, __int64); // rax
  __int64 v4; // rdi
  _QWORD *v5; // r13
  _QWORD *v6; // r12
  _QWORD *v7; // rdi
  _QWORD *v8; // r13
  _QWORD *v9; // r12
  _QWORD *v10; // rdi

  *a1 = &unk_49DAD08;
  v3 = (void (__fastcall *)(char *, char *, __int64))a1[30];
  if ( v3 )
  {
    a2 = (char *)(a1 + 28);
    v3(a2, a2, 3);
  }
  v4 = a1[24];
  if ( v4 )
  {
    a2 = (char *)(a1[26] - v4);
    j_j___libc_free_0(v4, a2);
  }
  v5 = (_QWORD *)a1[21];
  v6 = (_QWORD *)a1[20];
  if ( v5 != v6 )
  {
    do
    {
      v7 = (_QWORD *)v6[1];
      *v6 = &unk_49DACE8;
      if ( v7 != v6 + 3 )
      {
        a2 = (char *)(v6[3] + 1LL);
        j_j___libc_free_0(v7, a2);
      }
      v6 += 6;
    }
    while ( v5 != v6 );
    v6 = (_QWORD *)a1[20];
  }
  if ( v6 )
  {
    a2 = (char *)(a1[22] - (_QWORD)v6);
    j_j___libc_free_0(v6, a2);
  }
  v8 = (_QWORD *)a1[18];
  v9 = (_QWORD *)a1[17];
  if ( v8 != v9 )
  {
    do
    {
      if ( (_QWORD *)*v9 != v9 + 2 )
      {
        a2 = (char *)(v9[2] + 1LL);
        j_j___libc_free_0(*v9, a2);
      }
      v9 += 4;
    }
    while ( v8 != v9 );
    v9 = (_QWORD *)a1[17];
  }
  if ( v9 )
  {
    a2 = (char *)(a1[19] - (_QWORD)v9);
    j_j___libc_free_0(v9, a2);
  }
  if ( !*((_BYTE *)a1 + 124) )
    _libc_free(a1[13], a2);
  v10 = (_QWORD *)a1[9];
  if ( v10 != a1 + 11 )
    _libc_free(v10, a2);
}
