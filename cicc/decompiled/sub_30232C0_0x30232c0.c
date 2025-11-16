// Function: sub_30232C0
// Address: 0x30232c0
//
__int64 __fastcall sub_30232C0(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // r13
  unsigned __int64 v6; // r13
  unsigned int v7; // ecx
  __int64 v8; // rsi
  unsigned int v9; // eax
  __int64 *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // edx
  int *v15; // rcx
  int v16; // esi
  unsigned int v17; // ebx
  unsigned __int8 *v18; // rax
  size_t v19; // rdx
  _QWORD *v20; // r8
  int v22; // ecx
  int v23; // edx
  int v24; // r9d
  int v25; // r9d
  size_t v26; // [rsp+8h] [rbp-78h]
  _QWORD v27[3]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v28; // [rsp+28h] [rbp-58h]
  void *dest; // [rsp+30h] [rbp-50h]
  __int64 v30; // [rsp+38h] [rbp-48h]
  __int64 v31; // [rsp+40h] [rbp-40h]

  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 1104) + 56LL) + 16LL * (a3 & 0x7FFFFFFF));
  v30 = 0x100000000LL;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  *(_BYTE *)(a1 + 16) = 0;
  v27[0] = &unk_49DD210;
  v31 = a1;
  v27[1] = 0;
  v27[2] = 0;
  v28 = 0;
  dest = 0;
  sub_CB5980((__int64)v27, 0, 0, 0);
  v7 = *(_DWORD *)(a2 + 1136);
  v8 = *(_QWORD *)(a2 + 1120);
  if ( v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v10 = (__int64 *)(v8 + 40LL * v9);
    v11 = *v10;
    if ( v6 == *v10 )
      goto LABEL_3;
    v23 = 1;
    while ( v11 != -4096 )
    {
      v24 = v23 + 1;
      v9 = (v7 - 1) & (v23 + v9);
      v10 = (__int64 *)(v8 + 40LL * v9);
      v11 = *v10;
      if ( v6 == *v10 )
        goto LABEL_3;
      v23 = v24;
    }
  }
  v10 = (__int64 *)(v8 + 40LL * v7);
LABEL_3:
  v12 = *((unsigned int *)v10 + 8);
  v13 = v10[2];
  if ( (_DWORD)v12 )
  {
    v14 = (v12 - 1) & (37 * a3);
    v15 = (int *)(v13 + 8LL * v14);
    v16 = *v15;
    if ( *v15 == a3 )
      goto LABEL_5;
    v22 = 1;
    while ( v16 != -1 )
    {
      v25 = v22 + 1;
      v14 = (v12 - 1) & (v22 + v14);
      v15 = (int *)(v13 + 8LL * v14);
      v16 = *v15;
      if ( *v15 == a3 )
        goto LABEL_5;
      v22 = v25;
    }
  }
  v15 = (int *)(v13 + 8 * v12);
LABEL_5:
  v17 = v15[1];
  v18 = (unsigned __int8 *)sub_3058FE0(v6);
  if ( v19 > v28 - (__int64)dest )
  {
    v20 = (_QWORD *)sub_CB6200((__int64)v27, v18, v19);
  }
  else
  {
    v20 = v27;
    if ( v19 )
    {
      v26 = v19;
      memcpy(dest, v18, v19);
      v20 = v27;
      dest = (char *)dest + v26;
    }
  }
  sub_CB59D0((__int64)v20, v17);
  v27[0] = &unk_49DD210;
  sub_CB5840((__int64)v27);
  return a1;
}
