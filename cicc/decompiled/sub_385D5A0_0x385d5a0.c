// Function: sub_385D5A0
// Address: 0x385d5a0
//
void __fastcall sub_385D5A0(unsigned __int64 a1)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 *v5; // rdi

  v2 = *(unsigned int *)(a1 + 184);
  *(_QWORD *)a1 = &unk_4A3DE10;
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 168);
    v4 = &v3[2 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = (unsigned __int64 *)v3[1];
        if ( v5 )
          sub_385CEA0(v5);
      }
      v3 += 2;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  *(_QWORD *)a1 = &unk_49EE078;
  sub_16366C0((_QWORD *)a1);
  j_j___libc_free_0(a1);
}
