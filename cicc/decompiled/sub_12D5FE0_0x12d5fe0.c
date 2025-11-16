// Function: sub_12D5FE0
// Address: 0x12d5fe0
//
__int64 __fastcall sub_12D5FE0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rax

  v2 = *(_QWORD *)(a1 + 176);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 192) - v2);
  v3 = *(_QWORD *)(a1 + 152);
  if ( v3 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 168) - v3);
  v4 = *(unsigned int *)(a1 + 136);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 120);
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
          return j___libc_free_0(*(_QWORD *)(a1 + 120));
      }
      v8 = *(_QWORD *)(v5 + 24);
      v5 += 40;
      j_j___libc_free_0(v7, v8 + 1);
    }
    while ( v6 != v5 );
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 120));
}
