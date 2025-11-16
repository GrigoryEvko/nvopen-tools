// Function: sub_23AF1F0
// Address: 0x23af1f0
//
void __fastcall sub_23AF1F0(unsigned __int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 *v7; // r14
  __int64 v8; // rax
  _QWORD *v9; // rdi

  *(_QWORD *)a1 = &unk_4A161A0;
  v2 = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 56);
    v4 = v3 + 40 * v2;
    do
    {
      if ( *(_QWORD *)v3 != -8192 && *(_QWORD *)v3 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v3 + 16), 16LL * *(unsigned int *)(v3 + 32), 8);
      v3 += 40;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 72);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 56), 40 * v2, 8);
  if ( *(_BYTE *)(a1 + 40) )
  {
    v5 = *(unsigned int *)(a1 + 32);
    *(_BYTE *)(a1 + 40) = 0;
    if ( (_DWORD)v5 )
    {
      v6 = *(__int64 **)(a1 + 16);
      v7 = &v6[5 * v5];
      do
      {
        while ( 1 )
        {
          if ( *v6 <= 0x7FFFFFFFFFFFFFFDLL )
          {
            v6[1] = (__int64)&unk_49DB368;
            v8 = v6[4];
            if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
              break;
          }
          v6 += 5;
          if ( v7 == v6 )
            goto LABEL_17;
        }
        v9 = v6 + 2;
        v6 += 5;
        sub_BD60C0(v9);
      }
      while ( v7 != v6 );
LABEL_17:
      v5 = *(unsigned int *)(a1 + 32);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 40 * v5, 8);
  }
  j_j___libc_free_0(a1);
}
