// Function: sub_BEFBE0
// Address: 0xbefbe0
//
void __fastcall sub_BEFBE0(_QWORD *a1)
{
  _QWORD *v1; // r12
  __int64 (__fastcall *v2)(_QWORD *); // rax
  _QWORD *v3; // rdi

  v1 = (_QWORD *)*a1;
  if ( *a1 )
  {
    v2 = *(__int64 (__fastcall **)(_QWORD *))(*v1 + 8LL);
    if ( v2 == sub_BD9990 )
    {
      v3 = (_QWORD *)v1[1];
      *v1 = &unk_49DB390;
      if ( v3 != v1 + 3 )
        j_j___libc_free_0(v3, v1[3] + 1LL);
      j_j___libc_free_0(v1, 48);
    }
    else
    {
      v2((_QWORD *)*a1);
    }
  }
}
