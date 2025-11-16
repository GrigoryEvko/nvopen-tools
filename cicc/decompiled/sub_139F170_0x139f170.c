// Function: sub_139F170
// Address: 0x139f170
//
__int64 __fastcall sub_139F170(__int64 a1)
{
  bool v2; // zf
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rdi
  unsigned __int64 v8; // rdi

  v2 = *(_BYTE *)(a1 + 520) == 0;
  *(_QWORD *)a1 = &unk_49E9608;
  if ( !v2 )
  {
    v4 = *(unsigned int *)(a1 + 512);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD *)(a1 + 496);
      v6 = v5 + 24 * v4;
      do
      {
        if ( *(_QWORD *)v5 != -16 && *(_QWORD *)v5 != -8 && *(_DWORD *)(v5 + 16) > 0x40u )
        {
          v7 = *(_QWORD *)(v5 + 8);
          if ( v7 )
            j_j___libc_free_0_0(v7);
        }
        v5 += 24;
      }
      while ( v6 != v5 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 496));
    v8 = *(_QWORD *)(a1 + 208);
    if ( v8 != *(_QWORD *)(a1 + 200) )
      _libc_free(v8);
  }
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
