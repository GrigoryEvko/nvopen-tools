// Function: sub_97EB20
// Address: 0x97eb20
//
__int64 __fastcall sub_97EB20(__int64 a1)
{
  bool v2; // zf
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdi
  __int64 v10; // rax

  v2 = *(_BYTE *)(a1 + 400) == 0;
  *(_QWORD *)a1 = &unk_49D9670;
  if ( !v2 )
  {
    *(_BYTE *)(a1 + 400) = 0;
    v4 = *(_QWORD *)(a1 + 376);
    if ( v4 )
      j_j___libc_free_0(v4, *(_QWORD *)(a1 + 392) - v4);
    v5 = *(_QWORD *)(a1 + 352);
    if ( v5 )
      j_j___libc_free_0(v5, *(_QWORD *)(a1 + 368) - v5);
    v6 = *(unsigned int *)(a1 + 336);
    if ( (_DWORD)v6 )
    {
      v7 = *(_QWORD *)(a1 + 320);
      v8 = v7 + 40 * v6;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v7 <= 0xFFFFFFFD )
          {
            v9 = *(_QWORD *)(v7 + 8);
            if ( v9 != v7 + 24 )
              break;
          }
          v7 += 40;
          if ( v8 == v7 )
            goto LABEL_13;
        }
        v10 = *(_QWORD *)(v7 + 24);
        v7 += 40;
        j_j___libc_free_0(v9, v10 + 1);
      }
      while ( v8 != v7 );
LABEL_13:
      v6 = *(unsigned int *)(a1 + 336);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 320), 40 * v6, 8);
  }
  return sub_BB9280(a1);
}
