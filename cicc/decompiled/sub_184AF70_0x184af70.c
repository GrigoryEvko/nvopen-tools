// Function: sub_184AF70
// Address: 0x184af70
//
__int64 __fastcall sub_184AF70(__int64 *a1, __int64 *a2, int a3)
{
  __int64 v4; // r12
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r12

  if ( a3 == 1 )
  {
    *a1 = *a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v4 = *a1;
      if ( *a1 )
      {
        v5 = *(_QWORD *)(v4 + 80);
        if ( v5 != v4 + 96 )
          _libc_free(v5);
        if ( (*(_BYTE *)(v4 + 8) & 1) == 0 )
          j___libc_free_0(*(_QWORD *)(v4 + 16));
        j_j___libc_free_0(v4, 160);
      }
    }
    return 0;
  }
  v6 = *a2;
  v7 = sub_22077B0(160);
  v8 = v7;
  if ( v7 )
    sub_184ACC0(v7, v6);
  *a1 = v8;
  return 0;
}
