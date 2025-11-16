// Function: sub_A73D10
// Address: 0xa73d10
//
void __fastcall sub_A73D10(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v3; // r14
  __int64 v4; // rax
  unsigned __int64 *v5; // r13
  __int64 v6; // r15
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rcx
  __int64 v10; // rax

  v3 = &a2[a3];
  if ( a2 != v3 )
  {
    v4 = *(unsigned int *)(a1 + 8);
    v5 = a2;
    v6 = a1 + 16;
    do
    {
      v7 = *v5;
      if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, v6, v4 + 1, 4);
        v4 = *(unsigned int *)(a1 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v4) = v7;
      v8 = HIDWORD(v7);
      v9 = *(unsigned int *)(a1 + 12);
      v10 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v10;
      if ( v10 + 1 > v9 )
      {
        sub_C8D5F0(a1, v6, v10 + 1, 4);
        v10 = *(unsigned int *)(a1 + 8);
      }
      ++v5;
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v10) = v8;
      v4 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v4;
    }
    while ( v3 != v5 );
  }
}
