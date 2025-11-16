// Function: sub_2FB54B0
// Address: 0x2fb54b0
//
__int64 __fastcall sub_2FB54B0(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // r13
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int64 *v6; // r8
  __int64 v7; // r15
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  unsigned __int64 v11; // r11
  int v12; // eax
  int v13; // r10d
  unsigned int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // rdx
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rax
  __int64 *v26; // [rsp+10h] [rbp-58h]
  _DWORD v27[2]; // [rsp+20h] [rbp-48h] BYREF
  __int64 v28; // [rsp+28h] [rbp-40h] BYREF
  __int64 v29; // [rsp+30h] [rbp-38h]

  v2 = 0;
  v3 = 0;
  v5 = sub_F03E60(2u, *((_DWORD *)a1 + 47), 9, 0, (__int64)v27, a2, 1u);
  v28 = 0;
  v6 = &v28;
  v7 = v5;
  v29 = 0;
  v8 = (unsigned __int64 *)a1[24];
  v9 = *v8;
  if ( !*v8 )
    goto LABEL_13;
LABEL_2:
  *v8 = *(_QWORD *)v9;
LABEL_3:
  memset((void *)v9, 0, 0xB8u);
  v10 = (_QWORD *)v9;
  do
  {
    *v10 = 0;
    v10 += 2;
    *(v10 - 1) = 0;
  }
  while ( v10 != (_QWORD *)(v9 + 144) );
  v11 = v9 & 0xFFFFFFFFFFFFFFC0LL;
  while ( 1 )
  {
    v12 = v27[v2];
    v13 = v12 + v3;
    if ( v12 + v3 != v3 )
    {
      v14 = v3;
      v15 = 0;
      do
      {
        v16 = v14++;
        v17 = &a1[2 * v16];
        *(_QWORD *)(v9 + 4 * v15) = *v17;
        *(_QWORD *)(v9 + 4 * v15 + 8) = v17[1];
        *(_DWORD *)(v9 + v15 + 144) = *((_DWORD *)a1 + v16 + 36);
        v15 += 4;
      }
      while ( v13 != v14 );
      v12 = v27[v2];
      v3 += v12;
    }
    v6[v2] = (unsigned int)(v12 - 1) | v11;
    if ( v2 == 1 )
      break;
    v8 = (unsigned __int64 *)a1[24];
    v2 = 1;
    v9 = *v8;
    if ( *v8 )
      goto LABEL_2;
LABEL_13:
    v24 = v8[1];
    v8[11] += 192LL;
    v11 = (v24 + 63) & 0xFFFFFFFFFFFFFFC0LL;
    if ( v8[2] < v11 + 192 || !v24 )
    {
      v26 = v6;
      v25 = sub_9D1E70((__int64)(v8 + 1), 192, 192, 6);
      v6 = v26;
      v9 = v25;
      goto LABEL_3;
    }
    v8[1] = v11 + 192;
    if ( v11 )
    {
      v9 = (v24 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      goto LABEL_3;
    }
  }
  *((_DWORD *)a1 + 46) = 1;
  memset(a1, 0, 0xB8u);
  memset(
    (void *)((unsigned __int64)(a1 + 2) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 16) & 0xFFFFFFF8) + 96) >> 3));
  a1[12] = 0;
  a1[22] = 0;
  memset(
    (void *)((unsigned __int64)(a1 + 13) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 104) & 0xFFFFFFF8) + 184) >> 3));
  v18 = (__int64 *)(v28 & 0xFFFFFFFFFFFFFFC0LL);
  v19 = *(_QWORD *)((v28 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v27[0] - 1) + 8);
  a1[1] = v28;
  v20 = v29;
  a1[12] = v19;
  v21 = *(_QWORD *)((v20 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v27[1] - 1) + 8);
  a1[2] = v20;
  a1[13] = v21;
  v22 = *v18;
  *((_DWORD *)a1 + 47) = 2;
  *a1 = v22;
  return v7;
}
