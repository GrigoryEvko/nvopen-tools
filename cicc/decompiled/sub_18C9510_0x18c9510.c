// Function: sub_18C9510
// Address: 0x18c9510
//
__int64 __fastcall sub_18C9510(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // r13
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  __int64 v6; // rax

  *(_QWORD *)a1 = off_49F2870;
  v2 = *(_QWORD *)(a1 + 360);
  if ( v2 != *(_QWORD *)(a1 + 352) )
    _libc_free(v2);
  v3 = *(unsigned int *)(a1 + 240);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 224);
    v5 = &v4[4 * v3];
    do
    {
      if ( *v4 != -16 && *v4 != -8 )
      {
        v6 = v4[3];
        if ( v6 != -8 && v6 != 0 && v6 != -16 )
          sub_1649B30(v4 + 1);
      }
      v4 += 4;
    }
    while ( v5 != v4 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 224));
  j___libc_free_0(*(_QWORD *)(a1 + 192));
  *(_QWORD *)a1 = &unk_49EE078;
  sub_16366C0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 448);
}
