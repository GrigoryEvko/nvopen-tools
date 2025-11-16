// Function: sub_20FCC80
// Address: 0x20fcc80
//
__int64 __fastcall sub_20FCC80(_QWORD *a1, int a2)
{
  __int64 v2; // r13
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int64 v6; // r10
  __int128 *v7; // r8
  __int64 v8; // r15
  unsigned __int64 v9; // rsi
  unsigned __int64 i; // r11
  int v11; // eax
  int v12; // r10d
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rdi
  _QWORD *v16; // rdx
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // r11
  __int64 v26; // rax
  int v27; // r9d
  unsigned __int64 v28; // r11
  __int64 v29; // rdx
  __int64 v30; // r10
  __int128 *v31; // [rsp+8h] [rbp-70h]
  __int128 *v32; // [rsp+8h] [rbp-70h]
  __int64 v33; // [rsp+10h] [rbp-68h]
  unsigned __int64 v34; // [rsp+10h] [rbp-68h]
  unsigned int v35; // [rsp+18h] [rbp-60h]
  __int64 v36; // [rsp+18h] [rbp-60h]
  unsigned __int64 v37; // [rsp+20h] [rbp-58h]
  __int64 v38; // [rsp+20h] [rbp-58h]
  _DWORD v39[2]; // [rsp+30h] [rbp-48h] BYREF
  __int128 v40; // [rsp+38h] [rbp-40h] BYREF

  v2 = 0;
  v3 = 0;
  v5 = sub_39461C0(2, *((_DWORD *)a1 + 49), 8, 0, (unsigned int)v39, a2, 1);
  v6 = a1[25];
  v7 = &v40;
  v8 = v5;
  v40 = 0;
  v9 = *(_QWORD *)v6;
  if ( !*(_QWORD *)v6 )
    goto LABEL_11;
LABEL_2:
  *(_QWORD *)v6 = *(_QWORD *)v9;
LABEL_3:
  memset((void *)v9, 0, 0xC0u);
  for ( i = v9 & 0xFFFFFFFFFFFFFFC0LL; ; i = 0 )
  {
    v11 = v39[v2];
    v12 = v11 + v3;
    if ( v11 + v3 != v3 )
    {
      v13 = v3;
      v14 = 0;
      do
      {
        v15 = v13++;
        v16 = &a1[2 * v15];
        *(_QWORD *)(v9 + 2 * v14) = *v16;
        *(_QWORD *)(v9 + 2 * v14 + 8) = v16[1];
        *(_QWORD *)(v9 + v14 + 128) = a1[v15 + 16];
        v14 += 8;
      }
      while ( v12 != v13 );
      v11 = v39[v2];
      v3 += v11;
    }
    *((_QWORD *)v7 + v2) = (unsigned int)(v11 - 1) | i;
    if ( v2 == 1 )
      break;
    v6 = a1[25];
    v2 = 1;
    v9 = *(_QWORD *)v6;
    if ( *(_QWORD *)v6 )
      goto LABEL_2;
LABEL_11:
    v23 = *(_QWORD *)(v6 + 8);
    v24 = *(_QWORD *)(v6 + 16);
    *(_QWORD *)(v6 + 88) += 192LL;
    if ( ((v23 + 63) & 0xFFFFFFFFFFFFFFC0LL) - v23 + 192 <= v24 - v23 )
    {
      v9 = (v23 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      *(_QWORD *)(v6 + 8) = v9 + 192;
    }
    else
    {
      v31 = v7;
      v33 = v6;
      v35 = *(_DWORD *)(v6 + 32);
      v25 = 4096LL << (v35 >> 7);
      if ( v35 >> 7 >= 0x1E )
        v25 = 0x40000000000LL;
      v37 = v25;
      v26 = malloc(v25);
      v28 = v37;
      v29 = v35;
      v30 = v33;
      v7 = v31;
      if ( !v26 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v30 = v33;
        v7 = v31;
        v28 = v37;
        v26 = 0;
        v29 = *(unsigned int *)(v33 + 32);
      }
      if ( *(_DWORD *)(v30 + 36) <= (unsigned int)v29 )
      {
        v32 = v7;
        v34 = v28;
        v36 = v26;
        v38 = v30;
        sub_16CD150(v30 + 24, (const void *)(v30 + 40), 0, 8, (int)v7, v27);
        v30 = v38;
        v7 = v32;
        v28 = v34;
        v26 = v36;
        v29 = *(unsigned int *)(v38 + 32);
      }
      v9 = (v26 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      *(_QWORD *)(*(_QWORD *)(v30 + 24) + 8 * v29) = v26;
      ++*(_DWORD *)(v30 + 32);
      *(_QWORD *)(v30 + 16) = v26 + v28;
      *(_QWORD *)(v30 + 8) = v9 + 192;
    }
    if ( v9 )
      goto LABEL_3;
  }
  *((_DWORD *)a1 + 48) = 1;
  memset(a1, 0, 0xB8u);
  v17 = (__int64 *)(v40 & 0xFFFFFFFFFFFFFFC0LL);
  v18 = *(_QWORD *)((v40 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v39[0] - 1) + 8);
  a1[1] = v40;
  v19 = *((_QWORD *)&v40 + 1);
  a1[12] = v18;
  v20 = *(_QWORD *)((v19 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v39[1] - 1) + 8);
  a1[2] = v19;
  a1[13] = v20;
  v21 = *v17;
  *((_DWORD *)a1 + 49) = 2;
  *a1 = v21;
  return v8;
}
