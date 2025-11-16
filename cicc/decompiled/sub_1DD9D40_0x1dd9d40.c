// Function: sub_1DD9D40
// Address: 0x1dd9d40
//
void __fastcall sub_1DD9D40(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r14
  int v3; // r8d
  int v4; // r9d
  unsigned __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 *v12; // rax
  __int64 v13; // rax
  unsigned __int64 *v14; // rax
  __int64 *v15; // r13
  __int64 v16; // rbx
  __int64 *v17; // r12
  unsigned __int64 v18; // rax
  __int64 *v19; // rbx
  __int64 *v20; // rdi
  __int64 v21; // r14
  unsigned __int64 *v22; // rdx
  unsigned __int64 v23; // rcx
  __int64 v24; // rax

  v2 = a2[1];
  if ( v2 == a2[7] + 320LL )
  {
    v21 = *(_QWORD *)(a1 + 336);
    v5 = sub_1DD57D0(a1, 0, 0);
    v7 = v21 & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1 + 336 == (*(_QWORD *)(a1 + 336) & 0xFFFFFFFFFFFFFFF8LL) )
      v22 = *(unsigned __int64 **)(a1 + 344);
    else
      v22 = *(unsigned __int64 **)(v7 + 8);
    v23 = *v22;
    v24 = *(_QWORD *)v5;
    *(_QWORD *)(v5 + 8) = v22;
    v8 = (__int64 *)(v5 & 0xFFFFFFFFFFFFFFF9LL);
    v23 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v5 = v23 | v24 & 7;
    *(_QWORD *)(v23 + 8) = v5;
    *v22 = v5 | *v22 & 7;
  }
  else
  {
    v5 = sub_1DD57D0(a1, 0, 0);
    v6 = *(unsigned int *)(v2 + 48);
    v7 = v5 & 0xFFFFFFFFFFFFFFF9LL;
    v8 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 392) + 16 * v6) & 0xFFFFFFFFFFFFFFF8LL);
    v9 = *(_QWORD *)v5 & 7LL;
    v10 = *v8;
    *(_QWORD *)(v5 + 8) = v8;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v5 = v10 | v9;
    *(_QWORD *)(v10 + 8) = v5;
    *v8 = v5 | *v8 & 7;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 392) + 16LL * *(int *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 48) + 8) = v7;
  v11 = *(unsigned int *)(a1 + 400);
  if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 404) )
  {
    sub_16CD150(a1 + 392, (const void *)(a1 + 408), 0, 16, v3, v4);
    v11 = *(unsigned int *)(a1 + 400);
  }
  v12 = (unsigned __int64 *)(*(_QWORD *)(a1 + 392) + 16 * v11);
  *v12 = v7;
  v12[1] = (unsigned __int64)v8;
  v13 = *(unsigned int *)(a1 + 544);
  ++*(_DWORD *)(a1 + 400);
  if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 548) )
  {
    sub_16CD150(a1 + 536, (const void *)(a1 + 552), 0, 16, v3, v4);
    v13 = *(unsigned int *)(a1 + 544);
  }
  v14 = (unsigned __int64 *)(*(_QWORD *)(a1 + 536) + 16 * v13);
  v14[1] = (unsigned __int64)a2;
  *v14 = v7;
  ++*(_DWORD *)(a1 + 544);
  sub_1F107E0(a1, v5);
  v15 = *(__int64 **)(a1 + 536);
  v16 = 2LL * *(unsigned int *)(a1 + 544);
  v17 = &v15[v16];
  if ( &v15[v16] != v15 )
  {
    _BitScanReverse64(&v18, (v16 * 8) >> 4);
    sub_1DD9AE0(v15, (unsigned __int64)&v15[v16], 2LL * (int)(63 - (v18 ^ 0x3F)));
    if ( (unsigned __int64)v16 <= 32 )
    {
      sub_1DD5370(v15, &v15[v16]);
    }
    else
    {
      v19 = v15 + 32;
      sub_1DD5370(v15, v15 + 32);
      if ( v17 != v15 + 32 )
      {
        do
        {
          v20 = v19;
          v19 += 2;
          sub_1DD5300(v20);
        }
        while ( v17 != v19 );
      }
    }
  }
}
