// Function: sub_1B67D30
// Address: 0x1b67d30
//
void *__fastcall sub_1B67D30(_QWORD *a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rdi

  v2 = a1 + 20;
  v3 = (_QWORD *)a1[20];
  *a1 = off_49F68B8;
  if ( v3 != a1 + 20 )
  {
    do
    {
      v4 = v3;
      v3 = (_QWORD *)*v3;
      v5 = v4[2];
      if ( v5 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
      j_j___libc_free_0(v4, 24);
    }
    while ( v3 != v2 );
  }
  return sub_1636790(a1);
}
