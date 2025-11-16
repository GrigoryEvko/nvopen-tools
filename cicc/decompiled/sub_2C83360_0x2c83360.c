// Function: sub_2C83360
// Address: 0x2c83360
//
void __fastcall sub_2C83360(_QWORD *a1)
{
  unsigned __int64 i; // rdi
  _QWORD *v3; // rbx
  __int64 v4; // rsi
  void (*v5)(); // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rdi
  __int64 v8; // [rsp-8h] [rbp-18h]

  *a1 = &unk_4A250F8;
  for ( i = a1[2]; a1[1] != i; i = a1[2] )
  {
    v3 = *(_QWORD **)(i - 8);
    v4 = *((unsigned int *)v3 + 2);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)*v3 + 104LL))(
      *v3,
      v4,
      *((unsigned int *)v3 + 3),
      v3[2],
      v3[3],
      *((unsigned __int8 *)v3 + 48),
      v3[4],
      v3[5]);
    if ( *((_BYTE *)v3 + 48) )
    {
      v5 = *(void (**)())(*(_QWORD *)*v3 + 96LL);
      if ( v5 != nullsub_20 )
        ((void (__fastcall *)(_QWORD, __int64, __int64))v5)(*v3, v4, v8);
    }
    v6 = a1[2];
    a1[2] = v6 - 8;
    v7 = *(_QWORD *)(v6 - 8);
    if ( v7 )
      j_j___libc_free_0(v7);
  }
  if ( i )
    j_j___libc_free_0(i);
}
