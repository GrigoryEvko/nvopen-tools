// Function: sub_16206C0
// Address: 0x16206c0
//
__int64 *__fastcall sub_16206C0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  void *v4; // r13
  __int64 v5; // rax
  size_t v6; // rdx
  void *v7; // r10
  __int64 v8; // rax
  size_t v9; // rdx
  void *v10; // r8
  __int64 v11; // rax
  size_t v12; // rdx
  size_t v13; // r11
  __int64 *v14; // r15
  size_t v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  size_t v18; // rdx
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rcx
  void *v23; // [rsp+0h] [rbp-A0h]
  size_t v24; // [rsp+8h] [rbp-98h]
  size_t v25; // [rsp+10h] [rbp-90h]
  size_t v26; // [rsp+10h] [rbp-90h]
  size_t v27; // [rsp+18h] [rbp-88h]
  void *v28; // [rsp+18h] [rbp-88h]
  void *v29; // [rsp+18h] [rbp-88h]
  __int64 v30; // [rsp+20h] [rbp-80h]
  __int64 v31; // [rsp+28h] [rbp-78h]
  __int64 v32; // [rsp+30h] [rbp-70h]
  __int64 v33; // [rsp+38h] [rbp-68h]
  __int64 v34; // [rsp+40h] [rbp-60h]
  int v35; // [rsp+48h] [rbp-58h]
  int v36; // [rsp+4Ch] [rbp-54h]
  void *v37; // [rsp+50h] [rbp-50h]
  int v38; // [rsp+50h] [rbp-50h]
  __int64 v39; // [rsp+58h] [rbp-48h]
  char v40; // [rsp+60h] [rbp-40h]
  char v41; // [rsp+64h] [rbp-3Ch]
  char v42; // [rsp+68h] [rbp-38h]
  char v43; // [rsp+6Ch] [rbp-34h]

  v2 = a2;
  v43 = *(_BYTE *)(a2 + 50);
  v35 = *(_DWORD *)(a2 + 36);
  v42 = *(_BYTE *)(a2 + 49);
  v41 = *(_BYTE *)(a2 + 48);
  v39 = *(_QWORD *)(a2 + 40);
  v3 = *(unsigned int *)(a2 + 8);
  v30 = *(_QWORD *)(a2 + 8 * (8 - v3));
  v31 = *(_QWORD *)(a2 + 8 * (7 - v3));
  v32 = *(_QWORD *)(a2 + 8 * (6 - v3));
  v33 = *(_QWORD *)(a2 + 8 * (5 - v3));
  v4 = *(void **)(a2 + 8 * (3 - v3));
  v34 = *(_QWORD *)(a2 + 8 * (4 - v3));
  if ( v4 )
  {
    v5 = sub_161E970((__int64)v4);
    v27 = v6;
    v4 = (void *)v5;
    v3 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v27 = 0;
  }
  v7 = *(void **)(a2 + 8 * (2 - v3));
  v36 = *(_DWORD *)(a2 + 32);
  if ( v7 )
  {
    v8 = sub_161E970((__int64)v7);
    v25 = v9;
    v7 = (void *)v8;
    v3 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v25 = 0;
  }
  v10 = *(void **)(a2 + 8 * (1 - v3));
  v40 = *(_BYTE *)(a2 + 28);
  if ( v10 )
  {
    v37 = v7;
    v11 = sub_161E970((__int64)v10);
    v7 = v37;
    v10 = (void *)v11;
    v13 = v12;
  }
  else
  {
    v13 = 0;
  }
  if ( *(_BYTE *)a2 != 15 )
    v2 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  v38 = *(_DWORD *)(a2 + 24);
  v14 = (__int64 *)(*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 16) & 4) != 0 )
    v14 = (__int64 *)*v14;
  v15 = v27;
  v16 = 0;
  if ( v27 )
  {
    v23 = v7;
    v24 = v13;
    v28 = v10;
    v17 = sub_161FF10(v14, v4, v15);
    v7 = v23;
    v13 = v24;
    v10 = v28;
    v16 = v17;
  }
  v18 = v25;
  v19 = 0;
  if ( v25 )
  {
    v26 = v13;
    v29 = v10;
    v20 = sub_161FF10(v14, v7, v18);
    v13 = v26;
    v10 = v29;
    v19 = v20;
  }
  v21 = 0;
  if ( v13 )
    v21 = sub_161FF10(v14, v10, v13);
  *a1 = sub_15B0DC0((int)v14, v38, v2, v21, v40, v19, v36, v16, v35, v34, v33, v32, v31, v30, v39, v41, v42, v43, 2);
  return a1;
}
