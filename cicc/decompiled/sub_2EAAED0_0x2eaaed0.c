// Function: sub_2EAAED0
// Address: 0x2eaaed0
//
_QWORD *__fastcall sub_2EAAED0(__int64 a1)
{
  _QWORD *result; // rax
  unsigned __int64 v3; // rdi
  void (*v4)(void); // rax

  result = *(_QWORD **)(a1 + 56);
  if ( !result )
  {
    result = (_QWORD *)sub_22077B0(8u);
    if ( result )
      *result = &unk_4A29780;
    v3 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 56) = result;
    if ( v3 )
    {
      v4 = *(void (**)(void))(*(_QWORD *)v3 + 8LL);
      if ( (char *)v4 == (char *)sub_2EAAD20 )
        j_j___libc_free_0(v3);
      else
        v4();
      return *(_QWORD **)(a1 + 56);
    }
  }
  return result;
}
