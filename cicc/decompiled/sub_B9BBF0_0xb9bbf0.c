// Function: sub_B9BBF0
// Address: 0xb9bbf0
//
_QWORD *__fastcall sub_B9BBF0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r14
  size_t v3; // rdx
  size_t v4; // rdx
  size_t v5; // r13
  size_t v6; // rdx
  __int64 v7; // rax
  size_t v8; // r9
  unsigned __int8 v9; // al
  size_t v10; // rdx
  __int64 *v11; // rdx
  __int64 *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r9
  size_t v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  const void *v21; // [rsp+0h] [rbp-70h]
  const void *v22; // [rsp+8h] [rbp-68h]
  const void *v23; // [rsp+10h] [rbp-60h]
  const void *v24; // [rsp+18h] [rbp-58h]
  size_t v25; // [rsp+20h] [rbp-50h]
  size_t v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+28h] [rbp-48h]
  unsigned int v29; // [rsp+30h] [rbp-40h]
  int v30; // [rsp+34h] [rbp-3Ch]
  size_t v31; // [rsp+38h] [rbp-38h]
  __int64 v32; // [rsp+38h] [rbp-38h]

  v2 = a2;
  v29 = (unsigned int)*(char *)(a2 + 1) >> 31;
  v30 = *(_DWORD *)(a2 + 4);
  v24 = (const void *)sub_A547D0(a2, 5);
  v31 = v3;
  v23 = (const void *)sub_A547D0(a2, 4);
  v5 = v4;
  v22 = (const void *)sub_A547D0(a2, 3);
  v26 = v6;
  v7 = sub_A547D0(a2, 2);
  v8 = v31;
  v21 = (const void *)v7;
  v9 = *(_BYTE *)(a2 - 16);
  v25 = v10;
  if ( (v9 & 2) != 0 )
  {
    v11 = *(__int64 **)(a2 - 32);
    v32 = v11[1];
    if ( *(_BYTE *)a2 == 16 )
      goto LABEL_4;
    goto LABEL_3;
  }
  v11 = (__int64 *)(a2 - 16 - 8LL * ((v9 >> 2) & 0xF));
  v32 = v11[1];
  if ( *(_BYTE *)a2 != 16 )
LABEL_3:
    v2 = *v11;
LABEL_4:
  v12 = (__int64 *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 8) & 4) != 0 )
    v12 = (__int64 *)*v12;
  v13 = 0;
  if ( v8 )
    v13 = sub_B9B140(v12, v24, v8);
  v14 = 0;
  if ( v5 )
    v14 = sub_B9B140(v12, v23, v5);
  v15 = v26;
  v16 = 0;
  if ( v26 )
  {
    v27 = v14;
    v17 = sub_B9B140(v12, v22, v15);
    v14 = v27;
    v16 = v17;
  }
  v18 = 0;
  if ( v25 )
  {
    v28 = v14;
    v19 = sub_B9B140(v12, v21, v25);
    v14 = v28;
    v18 = v19;
  }
  *a1 = sub_B0A3E0(v12, v2, v32, v18, v16, v14, v13, v30, v29, 2u, 1);
  return a1;
}
