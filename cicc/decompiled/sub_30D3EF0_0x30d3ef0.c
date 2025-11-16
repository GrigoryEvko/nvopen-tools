// Function: sub_30D3EF0
// Address: 0x30d3ef0
//
__int64 __fastcall sub_30D3EF0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 *v10; // rdx
  __int64 v11; // r9
  int v12; // r12d
  int v13; // r13d
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // ecx
  __int64 *v27; // rdx
  __int64 v28; // r9
  __int64 v29; // r12
  int v31; // edx
  __int64 v32; // rax
  int v33; // edx
  int v34; // r10d
  int v35; // r11d
  int v36; // [rsp+8h] [rbp-38h]
  int v37; // [rsp+Ch] [rbp-34h]

  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_QWORD *)(v6 + 680);
  v8 = *(unsigned int *)(v6 + 696);
  if ( !(_DWORD)v8 )
    goto LABEL_13;
  v9 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 24LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    v31 = 1;
    while ( v11 != -4096 )
    {
      v35 = v31 + 1;
      v9 = (v8 - 1) & (v31 + v9);
      v10 = (__int64 *)(v7 + 24LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v31 = v35;
    }
    goto LABEL_13;
  }
LABEL_3:
  if ( v10 == (__int64 *)(v7 + 24 * v8) )
  {
LABEL_13:
    sub_904010((__int64)a3, "; No analysis for the instruction");
    goto LABEL_6;
  }
  v12 = *((_DWORD *)v10 + 3);
  v13 = *((_DWORD *)v10 + 5);
  v36 = *((_DWORD *)v10 + 2);
  v37 = *((_DWORD *)v10 + 4);
  v14 = sub_904010((__int64)a3, "; cost before = ");
  v15 = sub_CB59F0(v14, v36);
  v16 = sub_904010(v15, ", cost after = ");
  v17 = sub_CB59F0(v16, v12);
  v18 = sub_904010(v17, ", threshold before = ");
  v19 = sub_CB59F0(v18, v37);
  v20 = sub_904010(v19, ", threshold after = ");
  v21 = sub_CB59F0(v20, v13);
  sub_904010(v21, ", ");
  v22 = sub_904010((__int64)a3, "cost delta = ");
  sub_CB59F0(v22, v12 - v36);
  if ( v37 != v13 )
  {
    v32 = sub_904010((__int64)a3, ", threshold delta = ");
    sub_CB59F0(v32, v13 - v37);
  }
LABEL_6:
  v23 = *(_QWORD *)(a1 + 8);
  v24 = *(_QWORD *)(v23 + 144);
  v25 = *(unsigned int *)(v23 + 160);
  if ( (_DWORD)v25 )
  {
    v26 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v27 = (__int64 *)(v24 + 16LL * v26);
    v28 = *v27;
    if ( a2 == *v27 )
    {
LABEL_8:
      if ( v27 != (__int64 *)(v24 + 16 * v25) )
      {
        v29 = v27[1];
        sub_904010((__int64)a3, ", simplified to ");
        sub_A69870(v29, a3, 1);
      }
    }
    else
    {
      v33 = 1;
      while ( v28 != -4096 )
      {
        v34 = v33 + 1;
        v26 = (v25 - 1) & (v33 + v26);
        v27 = (__int64 *)(v24 + 16LL * v26);
        v28 = *v27;
        if ( a2 == *v27 )
          goto LABEL_8;
        v33 = v34;
      }
    }
  }
  return sub_904010((__int64)a3, "\n");
}
