// Function: sub_2D2A990
// Address: 0x2d2a990
//
__int64 __fastcall sub_2D2A990(_DWORD *a1, unsigned int a2)
{
  __int64 v2; // r13
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int64 *v6; // r8
  __int64 v7; // r15
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // r10
  int v11; // eax
  int v12; // edi
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdx
  int v17; // esi
  _DWORD *v18; // rax
  int v19; // ecx
  __int64 v20; // rdx
  int v21; // ecx
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  __int64 *v25; // [rsp+10h] [rbp-58h]
  _DWORD v26[2]; // [rsp+20h] [rbp-48h] BYREF
  __int64 v27; // [rsp+28h] [rbp-40h] BYREF
  __int64 v28; // [rsp+30h] [rbp-38h]

  v2 = 0;
  v3 = 0;
  v5 = sub_F03E60(2u, a1[49], 16, 0, (__int64)v26, a2, 1u);
  v27 = 0;
  v6 = &v27;
  v7 = v5;
  v28 = 0;
  v8 = (unsigned __int64 *)*((_QWORD *)a1 + 25);
  v9 = *v8;
  if ( !*v8 )
    goto LABEL_11;
LABEL_2:
  *v8 = *(_QWORD *)v9;
LABEL_3:
  *(_QWORD *)v9 = 0;
  v10 = v9 & 0xFFFFFFFFFFFFFFC0LL;
  *(_QWORD *)(v9 + 184) = 0;
  memset(
    (void *)((v9 + 8) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v9 - (((_DWORD)v9 + 8) & 0xFFFFFFF8) + 192) >> 3));
  while ( 1 )
  {
    v11 = v26[v2];
    v12 = v11 + v3;
    if ( v3 != v11 + v3 )
    {
      v13 = v3;
      v14 = 0;
      do
      {
        v15 = v13++;
        *(_DWORD *)(v9 + 2 * v14) = a1[2 * v15];
        *(_DWORD *)(v9 + 2 * v14 + 4) = a1[2 * v15 + 1];
        *(_DWORD *)(v9 + v14 + 128) = a1[v15 + 32];
        v14 += 4;
      }
      while ( v12 != v13 );
      v11 = v26[v2];
      v3 += v11;
    }
    v6[v2] = (unsigned int)(v11 - 1) | v10;
    if ( v2 == 1 )
      break;
    v8 = (unsigned __int64 *)*((_QWORD *)a1 + 25);
    v2 = 1;
    v9 = *v8;
    if ( *v8 )
      goto LABEL_2;
LABEL_11:
    v23 = v8[1];
    v8[11] += 192LL;
    v10 = (v23 + 63) & 0xFFFFFFFFFFFFFFC0LL;
    if ( v8[2] < v10 + 192 || !v23 )
    {
      v25 = v6;
      v24 = sub_9D1E70((__int64)(v8 + 1), 192, 192, 6);
      v6 = v25;
      v9 = v24;
      goto LABEL_3;
    }
    v8[1] = v10 + 192;
    if ( v10 )
    {
      v9 = (v23 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      goto LABEL_3;
    }
  }
  v16 = v27;
  a1[48] = 1;
  v17 = v26[1];
  memset(a1, 0, 0xC0u);
  v18 = (_DWORD *)(v16 & 0xFFFFFFFFFFFFFFC0LL);
  v19 = *(_DWORD *)((v16 & 0xFFFFFFFFFFFFFFC0LL) + 8LL * (unsigned int)(v26[0] - 1) + 4);
  *((_QWORD *)a1 + 1) = v16;
  v20 = v28;
  a1[32] = v19;
  v21 = *(_DWORD *)((v20 & 0xFFFFFFFFFFFFFFC0LL) + 8LL * (unsigned int)(v17 - 1) + 4);
  *((_QWORD *)a1 + 2) = v20;
  a1[33] = v21;
  LODWORD(v18) = *v18;
  a1[49] = 2;
  *a1 = (_DWORD)v18;
  return v7;
}
