// Function: sub_23081A0
// Address: 0x23081a0
//
void __fastcall sub_23081A0(unsigned __int64 a1)
{
  bool v2; // zf
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned __int64 v6; // rdi

  v2 = *(_BYTE *)(a1 + 388) == 0;
  *(_QWORD *)a1 = &unk_4A0B088;
  if ( v2 )
    _libc_free(*(_QWORD *)(a1 + 368));
  v3 = *(unsigned int *)(a1 + 352);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 336);
    v5 = v4 + 24 * v3;
    do
    {
      if ( *(_QWORD *)v4 != -8192 && *(_QWORD *)v4 != -4096 && *(_DWORD *)(v4 + 16) > 0x40u )
      {
        v6 = *(_QWORD *)(v4 + 8);
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      v4 += 24;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 352);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 336), 24 * v3, 8);
  if ( !*(_BYTE *)(a1 + 68) )
    _libc_free(*(_QWORD *)(a1 + 48));
  j_j___libc_free_0(a1);
}
