// Function: sub_1F17AD0
// Address: 0x1f17ad0
//
void __fastcall sub_1F17AD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  unsigned __int64 v5; // r15
  __int64 v7; // r14
  __int64 v8; // rdi
  unsigned int v9; // esi
  __int64 v10; // rax
  __int64 v11; // r9
  unsigned int v12; // r13d
  unsigned __int64 v13; // rbx
  unsigned int v14; // r10d
  unsigned int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // r11
  _QWORD *v18; // rax
  __int64 v19; // r9
  unsigned __int64 *v20; // rsi
  unsigned int v21; // r14d
  unsigned __int64 v22; // rbx
  unsigned int v23; // edx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // r9
  __int64 v28; // r13
  unsigned __int64 *v29; // rsi
  __int64 v30; // [rsp+0h] [rbp-50h]
  int v31; // [rsp+Ch] [rbp-44h]
  _QWORD *v32; // [rsp+10h] [rbp-40h]

  v5 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = (a2 >> 1) & 3;
  v8 = a1 + 8;
  v9 = *(_DWORD *)(v8 + 8);
  v10 = *(_QWORD *)v8 + 16LL * (v9 - 1);
  v11 = *(_QWORD *)(*(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 12));
  v12 = *(_DWORD *)(*(_QWORD *)(v8 - 8) + 184LL) - v9;
  if ( v12 )
  {
    LODWORD(a5) = v7;
    do
    {
      v13 = v11 & 0xFFFFFFFFFFFFFFC0LL;
      v14 = a5 | *(_DWORD *)(v5 + 24);
      if ( (*(_DWORD *)((*(_QWORD *)((v11 & 0xFFFFFFFFFFFFFFC0LL) + 0x60) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (unsigned int)(*(__int64 *)((v11 & 0xFFFFFFFFFFFFFFC0LL) + 0x60) >> 1) & 3) > v14 )
      {
        v18 = (_QWORD *)(v11 & 0xFFFFFFFFFFFFFFC0LL);
        v16 = 0;
      }
      else
      {
        v15 = 0;
        do
        {
          v16 = ++v15;
          v17 = *(_QWORD *)((v11 & 0xFFFFFFFFFFFFFFC0LL) + 8LL * v15 + 0x60);
        }
        while ( (*(_DWORD *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v17 >> 1) & 3) <= v14 );
        v18 = (_QWORD *)(v13 + 8LL * v15);
      }
      v19 = (v16 << 32) | ((v11 & 0x3F) + 1);
      if ( *(_DWORD *)(a1 + 20) <= v9 )
      {
        v30 = v19;
        v31 = a5;
        v32 = v18;
        sub_16CD150(v8, (const void *)(a1 + 24), 0, 16, a5, v19);
        v9 = *(_DWORD *)(a1 + 16);
        v19 = v30;
        LODWORD(a5) = v31;
        v18 = v32;
      }
      v20 = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + 16LL * v9);
      *v20 = v13;
      v20[1] = v19;
      v9 = *(_DWORD *)(a1 + 16) + 1;
      *(_DWORD *)(a1 + 16) = v9;
      v11 = *v18;
      --v12;
    }
    while ( v12 );
  }
  v21 = *(_DWORD *)(v5 + 24) | v7;
  v22 = v11 & 0xFFFFFFFFFFFFFFC0LL;
  if ( v21 < (*(_DWORD *)((*(_QWORD *)((v11 & 0xFFFFFFFFFFFFFFC0LL) + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
            | (unsigned int)(*(__int64 *)((v11 & 0xFFFFFFFFFFFFFFC0LL) + 8) >> 1) & 3) )
  {
    v24 = 0;
  }
  else
  {
    v23 = 0;
    do
    {
      v24 = ++v23;
      v25 = *(_QWORD *)((v11 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * v23 + 8);
      v26 = v25 >> 1;
      a5 = v25 & 0xFFFFFFFFFFFFFFF8LL;
    }
    while ( v21 >= (*(_DWORD *)(a5 + 24) | (unsigned int)(v26 & 3)) );
  }
  v27 = v11 & 0x3F;
  v28 = (v24 << 32) | (v27 + 1);
  if ( *(_DWORD *)(a1 + 20) <= v9 )
  {
    sub_16CD150(v8, (const void *)(a1 + 24), 0, 16, a5, v27);
    v9 = *(_DWORD *)(a1 + 16);
  }
  v29 = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + 16LL * v9);
  *v29 = v22;
  v29[1] = v28;
  ++*(_DWORD *)(a1 + 16);
}
