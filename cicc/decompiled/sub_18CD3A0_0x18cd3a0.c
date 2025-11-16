// Function: sub_18CD3A0
// Address: 0x18cd3a0
//
__int64 __fastcall sub_18CD3A0(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rax

  v1 = *(unsigned int *)(a1 + 224);
  *(_QWORD *)a1 = off_49F29C0;
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 208);
    v3 = &v2[4 * v1];
    do
    {
      if ( *v2 != -16 && *v2 != -8 )
      {
        v4 = v2[3];
        if ( v4 != -8 && v4 != 0 && v4 != -16 )
          sub_1649B30(v2 + 1);
      }
      v2 += 4;
    }
    while ( v3 != v2 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 208));
  j___libc_free_0(*(_QWORD *)(a1 + 176));
  *(_QWORD *)a1 = &unk_49EE078;
  sub_16366C0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 352);
}
