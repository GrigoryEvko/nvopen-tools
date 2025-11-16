// Function: sub_B9B970
// Address: 0xb9b970
//
__int64 *__fastcall sub_B9B970(__int64 *a1, __int64 a2)
{
  size_t v2; // rdx
  size_t v3; // r14
  size_t v4; // rdx
  size_t v5; // rbx
  size_t v6; // rdx
  size_t v7; // rdx
  size_t v8; // rdx
  __int64 *v9; // r15
  __int64 v10; // r14
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rcx
  const void *v15; // [rsp+0h] [rbp-D0h]
  const void *v16; // [rsp+8h] [rbp-C8h]
  const void *v17; // [rsp+10h] [rbp-C0h]
  const void *v18; // [rsp+18h] [rbp-B8h]
  const void *v19; // [rsp+20h] [rbp-B0h]
  size_t v20; // [rsp+28h] [rbp-A8h]
  size_t v21; // [rsp+30h] [rbp-A0h]
  size_t v22; // [rsp+38h] [rbp-98h]
  __int64 v23; // [rsp+40h] [rbp-90h]
  __int64 v24; // [rsp+48h] [rbp-88h]
  __int64 v25; // [rsp+50h] [rbp-80h]
  __int64 v26; // [rsp+58h] [rbp-78h]
  __int64 v27; // [rsp+60h] [rbp-70h]
  int v28; // [rsp+68h] [rbp-68h]
  int v29; // [rsp+6Ch] [rbp-64h]
  int v30; // [rsp+70h] [rbp-60h]
  int v31; // [rsp+74h] [rbp-5Ch]
  __int64 v32; // [rsp+78h] [rbp-58h]
  char v33; // [rsp+80h] [rbp-50h]
  char v34; // [rsp+84h] [rbp-4Ch]
  unsigned __int8 v35; // [rsp+88h] [rbp-48h]
  char v36; // [rsp+8Ch] [rbp-44h]
  __int64 v37; // [rsp+90h] [rbp-40h]
  __int64 v38; // [rsp+98h] [rbp-38h]

  v37 = a2;
  v19 = (const void *)sub_A547D0(a2, 10);
  v3 = v2;
  v18 = (const void *)sub_A547D0(a2, 9);
  v5 = v4;
  v36 = *(_BYTE *)(a2 + 43);
  v28 = *(_DWORD *)(a2 + 36);
  v35 = *(_BYTE *)(a2 + 42);
  v34 = *(_BYTE *)(a2 + 41);
  v32 = *(_QWORD *)(a2 + 24);
  v23 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 8);
  v24 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 7);
  v25 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 6);
  v26 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 5);
  v27 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 4);
  v29 = *(_DWORD *)(a2 + 32);
  v17 = (const void *)sub_A547D0(a2, 3);
  v22 = v6;
  v30 = *(_DWORD *)(a2 + 20);
  v16 = (const void *)sub_A547D0(a2, 2);
  v21 = v7;
  v33 = *(_BYTE *)(a2 + 40);
  v15 = (const void *)sub_A547D0(a2, 1);
  v20 = v8;
  if ( *(_BYTE *)a2 != 16 )
    v37 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  v31 = *(_DWORD *)(a2 + 16);
  v9 = (__int64 *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 8) & 4) != 0 )
    v9 = (__int64 *)*v9;
  v38 = 0;
  if ( v3 )
    v38 = sub_B9B140(v9, v19, v3);
  v10 = 0;
  if ( v5 )
    v10 = sub_B9B140(v9, v18, v5);
  v11 = 0;
  if ( v22 )
    v11 = sub_B9B140(v9, v17, v22);
  v12 = 0;
  if ( v21 )
    v12 = sub_B9B140(v9, v16, v21);
  v13 = 0;
  if ( v20 )
    v13 = sub_B9B140(v9, v15, v20);
  *a1 = sub_AF30C0(
          (int)v9,
          v31,
          v37,
          v13,
          v33,
          v12,
          v30,
          v11,
          v29,
          v27,
          v26,
          v25,
          v24,
          v23,
          v32,
          v34,
          v35,
          v28,
          v36,
          v10,
          v38,
          2u);
  return a1;
}
