// Function: sub_1F14EF0
// Address: 0x1f14ef0
//
__int64 __fastcall sub_1F14EF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v4; // r12
  __int64 *v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned int v9; // r8d
  __int64 v10; // rdx
  __int64 *v11; // rdi
  __int64 v12; // r9
  int v13; // r11d
  unsigned int v14; // esi
  __int64 v15; // rax
  unsigned int v16; // r9d
  unsigned int v17; // eax

  v2 = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)v2 )
    return 0;
  v4 = 3 * v2;
  v5 = *(__int64 **)a2;
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL);
  v7 = sub_1DA9310(v6, **(_QWORD **)a2);
  v8 = *(_QWORD *)(v6 + 392);
  v9 = 1;
  v10 = v7;
  v11 = &v5[v4];
  v12 = *(_QWORD *)(v8 + 16LL * *(unsigned int *)(v7 + 48) + 8);
  v13 = *(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  v14 = *(_DWORD *)((v5[v4 - 2] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v5[v4 - 2] >> 1) & 3;
  if ( (v13 | (unsigned int)(v12 >> 1) & 3) < v14 )
  {
    do
    {
      v15 = v5[1];
      v16 = v13 | (v12 >> 1) & 3;
      while ( (*(_DWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v15 >> 1) & 3) <= v16 )
      {
        v15 = v5[4];
        v5 += 3;
      }
      if ( v11 == v5 )
        break;
      do
      {
        v10 = *(_QWORD *)(v10 + 8);
        v12 = *(_QWORD *)(v8 + 16LL * *(unsigned int *)(v10 + 48) + 8);
        v13 = *(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v17 = v13 | (v12 >> 1) & 3;
      }
      while ( v17 <= ((unsigned int)(*v5 >> 1) & 3 | *(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24)) );
      ++v9;
    }
    while ( v14 > v17 );
  }
  return v9;
}
