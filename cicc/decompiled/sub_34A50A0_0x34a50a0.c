// Function: sub_34A50A0
// Address: 0x34a50a0
//
__int64 __fastcall sub_34A50A0(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // r13
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int64 *v6; // r8
  __int64 v7; // r15
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r11
  int v11; // eax
  int v12; // r10d
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rcx
  unsigned int v15; // edx
  __int64 v16; // rsi
  _QWORD *v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v24; // rsi
  __int64 v25; // rax
  __int64 *v26; // [rsp+10h] [rbp-58h]
  _DWORD v27[2]; // [rsp+20h] [rbp-48h] BYREF
  __int64 v28; // [rsp+28h] [rbp-40h] BYREF
  __int64 v29; // [rsp+30h] [rbp-38h]

  v2 = 0;
  v3 = 0;
  v5 = sub_F03E60(2u, *((_DWORD *)a1 + 49), 11, 0, (__int64)v27, a2, 1u);
  v28 = 0;
  v6 = &v28;
  v7 = v5;
  v29 = 0;
  v8 = (unsigned __int64 *)a1[25];
  v9 = *v8;
  if ( !*v8 )
    goto LABEL_11;
LABEL_2:
  *v8 = *(_QWORD *)v9;
LABEL_3:
  memset((void *)v9, 0, 0xC0u);
  v10 = v9 & 0xFFFFFFFFFFFFFFC0LL;
  while ( 1 )
  {
    v11 = v27[v2];
    v12 = v11 + v3;
    if ( v3 != v11 + v3 )
    {
      v13 = v9;
      v14 = v9 + 176;
      v15 = v3;
      do
      {
        v16 = v15++;
        v13 += 16LL;
        ++v14;
        v17 = &a1[2 * v16];
        *(_QWORD *)(v13 - 16) = *v17;
        *(_QWORD *)(v13 - 8) = v17[1];
        *(_BYTE *)(v14 - 1) = *((_BYTE *)a1 + v16 + 176);
      }
      while ( v12 != v15 );
      v11 = v27[v2];
      v3 += v11;
    }
    v6[v2] = (unsigned int)(v11 - 1) | v10;
    if ( v2 == 1 )
      break;
    v8 = (unsigned __int64 *)a1[25];
    v2 = 1;
    v9 = *v8;
    if ( *v8 )
      goto LABEL_2;
LABEL_11:
    v24 = v8[1];
    v8[11] += 192LL;
    v10 = (v24 + 63) & 0xFFFFFFFFFFFFFFC0LL;
    if ( v8[2] < v10 + 192 || !v24 )
    {
      v26 = v6;
      v25 = sub_9D1E70((__int64)(v8 + 1), 192, 192, 6);
      v6 = v26;
      v9 = v25;
      goto LABEL_3;
    }
    v8[1] = v10 + 192;
    if ( v10 )
    {
      v9 = (v24 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      goto LABEL_3;
    }
  }
  *((_DWORD *)a1 + 48) = 1;
  memset(a1, 0, 0xB8u);
  v18 = (__int64 *)(v28 & 0xFFFFFFFFFFFFFFC0LL);
  v19 = *(_QWORD *)((v28 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v27[0] - 1) + 8);
  a1[1] = v28;
  v20 = v29;
  a1[12] = v19;
  v21 = *(_QWORD *)((v20 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v27[1] - 1) + 8);
  a1[2] = v20;
  a1[13] = v21;
  v22 = *v18;
  *((_DWORD *)a1 + 49) = 2;
  *a1 = v22;
  return v7;
}
