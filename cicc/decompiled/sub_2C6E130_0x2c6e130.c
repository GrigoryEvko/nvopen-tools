// Function: sub_2C6E130
// Address: 0x2c6e130
//
__int64 __fastcall sub_2C6E130(_QWORD *a1)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r14
  unsigned __int64 v7; // rdi

  v2 = a1[22];
  *a1 = &unk_4A24FA8;
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 40);
    if ( v3 )
      j_j___libc_free_0(v3);
    v4 = *(unsigned int *)(v2 + 32);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD **)(v2 + 16);
      v6 = &v5[11 * v4];
      do
      {
        if ( *v5 != -8192 && *v5 != -4096 )
        {
          v7 = v5[2];
          if ( (_QWORD *)v7 != v5 + 4 )
            _libc_free(v7);
        }
        v5 += 11;
      }
      while ( v6 != v5 );
      v4 = *(unsigned int *)(v2 + 32);
    }
    sub_C7D6A0(*(_QWORD *)(v2 + 16), 88 * v4, 8);
    j_j___libc_free_0(v2);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
