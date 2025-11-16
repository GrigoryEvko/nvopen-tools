// Function: sub_12EC620
// Address: 0x12ec620
//
__int64 __fastcall sub_12EC620(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  _QWORD *v6; // rdi
  _QWORD *v7; // rdi
  __int64 v8; // rdi
  _QWORD *v9; // rdi

  v3 = a1 + 15;
  *(v3 - 15) = &unk_49E7460;
  *v3 = &unk_49EE9E8;
  sub_168FB40(v3);
  v4 = (_QWORD *)a1[56];
  while ( v4 != a1 + 56 )
  {
    v5 = v4;
    v4 = (_QWORD *)*v4;
    v6 = (_QWORD *)v5[2];
    if ( v6 != v5 + 4 )
      j_j___libc_free_0(v6, v5[4] + 1LL);
    a2 = 48;
    j_j___libc_free_0(v5, 48);
  }
  v7 = (_QWORD *)a1[38];
  if ( v7 != a1 + 40 )
    _libc_free(v7, a2);
  v8 = a1[35];
  a1[15] = &unk_49E6A18;
  j___libc_free_0(v8);
  v9 = (_QWORD *)a1[16];
  if ( v9 != a1 + 18 )
    _libc_free(v9, a2);
  return sub_1691870(a1 + 1);
}
