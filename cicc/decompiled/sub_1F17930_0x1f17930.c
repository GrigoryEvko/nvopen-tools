// Function: sub_1F17930
// Address: 0x1f17930
//
__int64 __fastcall sub_1F17930(_QWORD *a1, int a2)
{
  __int64 v2; // r13
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int128 *v6; // r9
  __int64 *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rsi
  int v10; // eax
  int v11; // r10d
  unsigned int v12; // ecx
  __int64 v13; // rax
  __int64 v14; // rdi
  _QWORD *v15; // rdx
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v22; // rax
  __int128 *v23; // [rsp+10h] [rbp-58h]
  _DWORD v24[2]; // [rsp+20h] [rbp-48h] BYREF
  __int128 v25; // [rsp+28h] [rbp-40h] BYREF

  v2 = 0;
  v3 = 0;
  v5 = sub_39461C0(2, *((_DWORD *)a1 + 47), 9, 0, (unsigned int)v24, a2, 1);
  v6 = &v25;
  v7 = (__int64 *)a1[24];
  v8 = v5;
  v25 = 0;
  v9 = *v7;
  if ( !*v7 )
    goto LABEL_10;
LABEL_2:
  *v7 = *(_QWORD *)v9;
  while ( 1 )
  {
    memset((void *)v9, 0, 0xB8u);
    v10 = v24[v2];
    v11 = v10 + v3;
    if ( v3 != v10 + v3 )
    {
      v12 = v3;
      v13 = 0;
      do
      {
        v14 = v12++;
        v15 = &a1[2 * v14];
        *(_QWORD *)(v9 + 4 * v13) = *v15;
        *(_QWORD *)(v9 + 4 * v13 + 8) = v15[1];
        *(_DWORD *)(v9 + v13 + 144) = *((_DWORD *)a1 + v14 + 36);
        v13 += 4;
      }
      while ( v11 != v12 );
      v10 = v24[v2];
      v3 += v10;
    }
    *((_QWORD *)v6 + v2) = (unsigned int)(v10 - 1) | v9 & 0xFFFFFFFFFFFFFFC0LL;
    if ( v2 == 1 )
      break;
    v7 = (__int64 *)a1[24];
    v2 = 1;
    v9 = *v7;
    if ( *v7 )
      goto LABEL_2;
LABEL_10:
    v23 = v6;
    v22 = sub_145CBF0(v7 + 1, 192, 64);
    v6 = v23;
    v9 = v22;
  }
  *((_DWORD *)a1 + 46) = 1;
  memset(a1, 0, 0xB8u);
  v16 = (__int64 *)(v25 & 0xFFFFFFFFFFFFFFC0LL);
  v17 = *(_QWORD *)((v25 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v24[0] - 1) + 8);
  a1[1] = v25;
  v18 = *((_QWORD *)&v25 + 1);
  a1[12] = v17;
  v19 = *(_QWORD *)((v18 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v24[1] - 1) + 8);
  a1[2] = v18;
  a1[13] = v19;
  v20 = *v16;
  *((_DWORD *)a1 + 47) = 2;
  *a1 = v20;
  return v8;
}
