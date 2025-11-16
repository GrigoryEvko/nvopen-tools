// Function: sub_334D340
// Address: 0x334d340
//
void __fastcall sub_334D340(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r8
  unsigned __int64 v9; // r12
  bool v10; // zf
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // [rsp+Ch] [rbp-34h]

  v6 = *(_QWORD *)(a2 + 40);
  v7 = v6 + 16LL * *(unsigned int *)(a2 + 48);
  if ( v6 != v7 )
  {
    v8 = a3;
    do
    {
      v9 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = (*(_DWORD *)(v9 + 220))-- == 1;
      if ( v10 && v9 != a1 + 72 )
      {
        *(_BYTE *)(v9 + 249) |= 2u;
        v11 = *(unsigned int *)(a1 + 640);
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 644) )
        {
          v14 = v8;
          sub_C8D5F0(a1 + 632, (const void *)(a1 + 648), v11 + 1, 8u, v8, a6);
          v11 = *(unsigned int *)(a1 + 640);
          v8 = v14;
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 632) + 8 * v11) = v9;
        ++*(_DWORD *)(a1 + 640);
      }
      if ( (*(_BYTE *)v6 & 6) == 0 )
      {
        v12 = *(unsigned int *)(v6 + 8);
        if ( (_DWORD)v12 )
        {
          v13 = *(_QWORD *)(a1 + 784);
          if ( !*(_QWORD *)(v13 + 8 * v12) )
          {
            ++*(_DWORD *)(a1 + 776);
            *(_QWORD *)(v13 + 8LL * *(unsigned int *)(v6 + 8)) = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
            *(_DWORD *)(*(_QWORD *)(a1 + 808) + 4LL * *(unsigned int *)(v6 + 8)) = v8;
          }
        }
      }
      v6 += 16;
    }
    while ( v7 != v6 );
  }
}
