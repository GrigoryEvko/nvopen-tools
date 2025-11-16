// Function: sub_2295B30
// Address: 0x2295b30
//
char __fastcall sub_2295B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int16 v7; // ax
  __int64 *v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  char v14; // al
  __int64 v16; // rbx
  char *v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 *v23; // rdi
  __int64 *v24; // rax
  __int64 v25; // r14
  char *v26; // rax
  __int64 v27; // r14
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 *v32; // rdi
  char *v33; // [rsp+8h] [rbp-58h]
  char *v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+20h] [rbp-40h]

  v7 = *(_WORD *)(a3 + 24);
  if ( *(_WORD *)(a2 + 24) != 8 )
  {
    if ( v7 == 8 )
    {
      v24 = *(__int64 **)(a3 + 32);
      v25 = *v24;
      if ( *(_WORD *)(*v24 + 24) == 8 )
      {
        v10 = **(_QWORD **)(v25 + 32);
        v35 = sub_D33D80((_QWORD *)v25, *(_QWORD *)(a1 + 8), a3, a4, a5);
        v26 = *(char **)(v25 + 48);
        v27 = *(_QWORD *)(a1 + 8);
        v33 = v26;
        v31 = sub_D33D80((_QWORD *)a3, v27, v28, v29, v30);
        v32 = (__int64 *)v27;
        v9 = a2;
        v36 = (__int64)sub_DCAF50(v32, v31, 0);
        v34 = *(char **)(a3 + 48);
        v14 = sub_2294CC0(a1, v36, v35, a2, v10, v34, v33, a4);
        goto LABEL_4;
      }
    }
    goto LABEL_13;
  }
  v8 = *(__int64 **)(a2 + 32);
  if ( v7 != 8 )
  {
    v16 = *v8;
    if ( *(_WORD *)(*v8 + 24) == 8 )
    {
      v9 = **(_QWORD **)(v16 + 32);
      v36 = sub_D33D80((_QWORD *)*v8, *(_QWORD *)(a1 + 8), (__int64)v8, a4, a5);
      v17 = *(char **)(v16 + 48);
      v18 = *(_QWORD *)(a1 + 8);
      v34 = v17;
      v22 = sub_D33D80((_QWORD *)a2, v18, v19, v20, v21);
      v23 = (__int64 *)v18;
      v10 = a3;
      v35 = (__int64)sub_DCAF50(v23, v22, 0);
      v33 = *(char **)(a2 + 48);
      v14 = sub_2294CC0(a1, v36, v35, v9, a3, v34, v33, a4);
      goto LABEL_4;
    }
LABEL_13:
    BUG();
  }
  v9 = *v8;
  v36 = sub_D33D80((_QWORD *)a2, *(_QWORD *)(a1 + 8), (__int64)v8, a4, a5);
  v34 = *(char **)(a2 + 48);
  v10 = **(_QWORD **)(a3 + 32);
  v35 = sub_D33D80((_QWORD *)a3, *(_QWORD *)(a1 + 8), v11, v12, v13);
  v33 = *(char **)(a3 + 48);
  v14 = sub_2294CC0(a1, v36, v35, v9, v10, v34, v33, a4);
LABEL_4:
  if ( v14 || (unsigned __int8)sub_228FE90(a1, a2, a3, a4) )
    return 1;
  else
    return sub_228FB70(a1, v36, v35, v9, v10, v34, v33);
}
