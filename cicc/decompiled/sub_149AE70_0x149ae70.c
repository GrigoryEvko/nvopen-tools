// Function: sub_149AE70
// Address: 0x149ae70
//
__int64 __fastcall sub_149AE70(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rax

  *(_QWORD *)a1 = &unk_49EC9B0;
  v2 = *(_QWORD *)(a1 + 336);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 352) - v2);
  v3 = *(_QWORD *)(a1 + 312);
  if ( v3 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 328) - v3);
  v4 = *(unsigned int *)(a1 + 296);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 280);
    v6 = v5 + 40 * v4;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v5 <= 0xFFFFFFFD )
        {
          v7 = *(_QWORD *)(v5 + 8);
          if ( v7 != v5 + 24 )
            break;
        }
        v5 += 40;
        if ( v6 == v5 )
          goto LABEL_11;
      }
      v8 = *(_QWORD *)(v5 + 24);
      v5 += 40;
      j_j___libc_free_0(v7, v8 + 1);
    }
    while ( v6 != v5 );
  }
LABEL_11:
  j___libc_free_0(*(_QWORD *)(a1 + 280));
  sub_16367B0(a1);
  return j_j___libc_free_0(a1, 368);
}
