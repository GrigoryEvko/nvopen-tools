// Function: sub_1A48C40
// Address: 0x1a48c40
//
void *__fastcall sub_1A48C40(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi

  *(_QWORD *)a1 = off_49F53B0;
  v2 = *(unsigned int *)(a1 + 232);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 216);
    v4 = &v3[5 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( (_QWORD *)v5 != v3 + 3 )
          _libc_free(v5);
      }
      v3 += 5;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 216));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
