// Function: sub_B9BD90
// Address: 0xb9bd90
//
__int64 *__fastcall sub_B9BD90(__int64 *a1, __int64 a2)
{
  __int64 v4; // rbx
  int v5; // eax
  char v6; // di
  int v7; // r14d
  char v8; // si
  unsigned __int8 v9; // al
  _QWORD *v10; // rax
  __int64 v11; // r13
  size_t v12; // rdx
  const void *v13; // r11
  unsigned __int8 v14; // al
  size_t v15; // rdx
  size_t v16; // r8
  __int64 *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 *v20; // r15
  __int64 v21; // rcx
  const void *v22; // rsi
  size_t v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v28; // [rsp+0h] [rbp-70h]
  size_t v29; // [rsp+0h] [rbp-70h]
  size_t v30; // [rsp+8h] [rbp-68h]
  const void *v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+20h] [rbp-50h]
  char v36; // [rsp+2Ch] [rbp-44h]
  __int64 v37; // [rsp+30h] [rbp-40h]
  char v38; // [rsp+38h] [rbp-38h]
  int v39; // [rsp+3Ch] [rbp-34h]

  v4 = a2 - 16;
  v5 = *(_DWORD *)(a2 + 4);
  v6 = *(_BYTE *)(a2 + 21);
  v7 = *(_DWORD *)(a2 + 16);
  v8 = *(_BYTE *)(a2 + 20);
  v39 = v5;
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
  {
    v10 = *(_QWORD **)(a2 - 32);
    v33 = v10[8];
    v34 = v10[7];
    v37 = v10[6];
    v38 = v6;
    v36 = v8;
  }
  else
  {
    v38 = v6;
    v36 = v8;
    v10 = (_QWORD *)(v4 - 8LL * ((v9 >> 2) & 0xF));
    v33 = v10[8];
    v37 = v10[6];
    v34 = v10[7];
  }
  v35 = v10[3];
  v11 = v10[2];
  v28 = sub_A547D0(a2, 5);
  v30 = v12;
  v13 = (const void *)sub_A547D0(a2, 1);
  v14 = *(_BYTE *)(a2 - 16);
  v16 = v15;
  if ( (v14 & 2) != 0 )
    v17 = *(__int64 **)(a2 - 32);
  else
    v17 = (__int64 *)(v4 - 8LL * ((v14 >> 2) & 0xF));
  v18 = *(_QWORD *)(a2 + 8);
  v19 = *v17;
  v20 = (__int64 *)(v18 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v18 & 4) != 0 )
    v20 = (__int64 *)*v20;
  v21 = 0;
  if ( v30 )
  {
    v22 = (const void *)v28;
    v23 = v30;
    v29 = v16;
    v31 = v13;
    v24 = sub_B9B140(v20, v22, v23);
    v16 = v29;
    v13 = v31;
    v21 = v24;
  }
  v25 = 0;
  if ( v16 )
  {
    v32 = v21;
    v26 = sub_B9B140(v20, v13, v16);
    v21 = v32;
    v25 = v26;
  }
  *a1 = sub_B0B820(v20, v19, v25, v21, v11, v7, v35, v36, v38, v37, v34, v39, v33, 2u, 1);
  return a1;
}
