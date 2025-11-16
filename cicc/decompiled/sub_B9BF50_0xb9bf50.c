// Function: sub_B9BF50
// Address: 0xb9bf50
//
__int64 *__fastcall sub_B9BF50(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  const void *v5; // r13
  size_t v6; // rdx
  size_t v7; // r14
  size_t v8; // rdx
  unsigned __int8 v9; // al
  __int64 v10; // r15
  size_t v11; // rdx
  size_t v12; // r8
  __int64 *v13; // r15
  __int64 v14; // r9
  __int64 v15; // rax
  size_t v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v22; // rdx
  size_t v23; // [rsp+8h] [rbp-68h]
  size_t v24; // [rsp+8h] [rbp-68h]
  const void *v25; // [rsp+10h] [rbp-60h]
  const void *v26; // [rsp+18h] [rbp-58h]
  size_t v27; // [rsp+20h] [rbp-50h]
  __int64 v28; // [rsp+20h] [rbp-50h]
  __int64 v29; // [rsp+20h] [rbp-50h]
  __int64 v30; // [rsp+28h] [rbp-48h]
  int v31; // [rsp+30h] [rbp-40h]
  int v32; // [rsp+34h] [rbp-3Ch]
  __int64 v33; // [rsp+38h] [rbp-38h]

  v2 = a2 - 16;
  v31 = *(_DWORD *)(a2 + 20);
  v3 = *(_BYTE *)(a2 - 16);
  if ( (v3 & 2) != 0 )
  {
    v4 = *(_QWORD *)(a2 - 32);
    v5 = *(const void **)(v4 + 24);
    v33 = *(_QWORD *)(v4 + 32);
    if ( v5 )
    {
LABEL_3:
      v5 = (const void *)sub_B91420((__int64)v5);
      v7 = v6;
      goto LABEL_4;
    }
  }
  else
  {
    v22 = v2 - 8LL * ((v3 >> 2) & 0xF);
    v5 = *(const void **)(v22 + 24);
    v33 = *(_QWORD *)(v22 + 32);
    if ( v5 )
      goto LABEL_3;
  }
  v7 = 0;
LABEL_4:
  v26 = (const void *)sub_A547D0(a2, 2);
  v27 = v8;
  v32 = *(_DWORD *)(a2 + 16);
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
    v10 = *(_QWORD *)(a2 - 32);
  else
    v10 = v2 - 8LL * ((v9 >> 2) & 0xF);
  v30 = *(_QWORD *)(v10 + 8);
  v25 = (const void *)sub_A547D0(a2, 0);
  v12 = v11;
  v13 = (__int64 *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 8) & 4) != 0 )
    v13 = (__int64 *)*v13;
  v14 = 0;
  if ( v7 )
  {
    v23 = v11;
    v15 = sub_B9B140(v13, v5, v7);
    v12 = v23;
    v14 = v15;
  }
  v16 = v27;
  v17 = 0;
  if ( v27 )
  {
    v24 = v12;
    v28 = v14;
    v18 = sub_B9B140(v13, v26, v16);
    v12 = v24;
    v14 = v28;
    v17 = v18;
  }
  v19 = 0;
  if ( v12 )
  {
    v29 = v14;
    v20 = sub_B9B140(v13, v25, v12);
    v14 = v29;
    v19 = v20;
  }
  *a1 = sub_B0F520(v13, v19, v30, v32, v17, v14, v31, v33, 2u, 1);
  return a1;
}
