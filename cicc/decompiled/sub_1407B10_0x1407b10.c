// Function: sub_1407B10
// Address: 0x1407b10
//
__int64 __fastcall sub_1407B10(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rdi

  *(_QWORD *)a1 = off_49EAFC0;
  v2 = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 176);
    v4 = v3 + 168 * v2;
    do
    {
      if ( *(_QWORD *)v3 != -16 && *(_QWORD *)v3 != -8 )
      {
        v5 = *(_QWORD *)(v3 + 88);
        if ( v5 != v3 + 104 )
          _libc_free(v5);
        if ( (*(_BYTE *)(v3 + 16) & 1) == 0 )
          j___libc_free_0(*(_QWORD *)(v3 + 24));
      }
      v3 += 168;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 176));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
