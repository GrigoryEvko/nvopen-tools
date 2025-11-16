// Function: sub_217D630
// Address: 0x217d630
//
void *__fastcall sub_217D630(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rbx
  _QWORD *v3; // r13

  *(_QWORD *)a1 = off_4A031D0;
  j___libc_free_0(*(_QWORD *)(a1 + 360));
  j___libc_free_0(*(_QWORD *)(a1 + 328));
  v1 = *(unsigned int *)(a1 + 312);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 296);
    v3 = &v2[5 * v1];
    do
    {
      if ( *v2 != -16 && *v2 != -8 )
        j___libc_free_0(v2[2]);
      v2 += 5;
    }
    while ( v3 != v2 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 296));
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
