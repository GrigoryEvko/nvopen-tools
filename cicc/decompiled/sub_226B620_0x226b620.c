// Function: sub_226B620
// Address: 0x226b620
//
__int64 __fastcall sub_226B620(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  unsigned __int64 v7; // rdi

  v2 = *(_QWORD *)(a1 + 200);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 176);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(unsigned int *)(a1 + 160);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 144);
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
      v5 += 40;
      j_j___libc_free_0(v7);
    }
    while ( v6 != v5 );
LABEL_11:
    v4 = *(unsigned int *)(a1 + 160);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 144), 40 * v4, 8);
}
