// Function: sub_E66D20
// Address: 0xe66d20
//
void __fastcall sub_E66D20(__int64 a1)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 *v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 v13; // rsi

  v2 = *(__int64 **)(a1 + 64);
  v3 = &v2[2 * *(unsigned int *)(a1 + 72)];
  while ( v3 != v2 )
  {
    v4 = v2[1];
    v5 = *v2;
    v2 += 2;
    sub_C7D6A0(v5, v4, 16);
  }
  *(_DWORD *)(a1 + 72) = 0;
  v6 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v6 )
  {
    *(_QWORD *)(a1 + 80) = 0;
    v7 = *(__int64 **)(a1 + 16);
    v8 = *v7;
    v9 = &v7[v6];
    v10 = v7 + 1;
    *(_QWORD *)a1 = *v7;
    *(_QWORD *)(a1 + 8) = v8 + 4096;
    if ( v9 != v7 + 1 )
    {
      while ( 1 )
      {
        v11 = *v10;
        v12 = (unsigned int)(v10 - v7) >> 7;
        v13 = 4096LL << v12;
        if ( v12 >= 0x1E )
          v13 = 0x40000000000LL;
        ++v10;
        sub_C7D6A0(v11, v13, 16);
        if ( v9 == v10 )
          break;
        v7 = *(__int64 **)(a1 + 16);
      }
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
}
