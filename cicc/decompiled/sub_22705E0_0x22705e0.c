// Function: sub_22705E0
// Address: 0x22705e0
//
void __fastcall sub_22705E0(unsigned __int64 a1)
{
  bool v2; // zf
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  unsigned __int64 v8; // rdi

  v2 = *(_BYTE *)(a1 + 232) == 0;
  *(_QWORD *)a1 = &unk_4A089F8;
  if ( !v2 )
  {
    *(_BYTE *)(a1 + 232) = 0;
    v3 = *(_QWORD *)(a1 + 208);
    if ( v3 )
      j_j___libc_free_0(v3);
    v4 = *(_QWORD *)(a1 + 184);
    if ( v4 )
      j_j___libc_free_0(v4);
    v5 = *(unsigned int *)(a1 + 168);
    if ( (_DWORD)v5 )
    {
      v6 = *(_QWORD *)(a1 + 152);
      v7 = v6 + 40 * v5;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v6 <= 0xFFFFFFFD )
          {
            v8 = *(_QWORD *)(v6 + 8);
            if ( v8 != v6 + 24 )
              break;
          }
          v6 += 40;
          if ( v7 == v6 )
            goto LABEL_13;
        }
        v6 += 40;
        j_j___libc_free_0(v8);
      }
      while ( v7 != v6 );
LABEL_13:
      v5 = *(unsigned int *)(a1 + 168);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 152), 40 * v5, 8);
  }
  j_j___libc_free_0(a1);
}
