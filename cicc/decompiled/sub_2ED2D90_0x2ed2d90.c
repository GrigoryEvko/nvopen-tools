// Function: sub_2ED2D90
// Address: 0x2ed2d90
//
void __fastcall sub_2ED2D90(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _QWORD *a8)
{
  __int64 v8; // r12
  __int64 *v9; // r11
  __int64 v10; // rbx
  __int64 *v11; // r9
  __int64 v12; // r13
  __int64 *v13; // r15
  __int64 *v14; // rax
  char *v15; // r9
  int v16; // r11d
  char *v17; // r10
  __int64 v18; // r14
  int v19; // r9d
  __int64 *v20; // r10
  __int64 *v21; // rax
  __int64 *v22; // r10
  unsigned int v23; // ebx
  unsigned int v24; // eax
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 *v27; // rdi
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // r10
  unsigned __int64 v32; // rbx
  __int64 v33; // rax
  unsigned __int64 v34; // r14
  char v35; // al
  __int64 *v37; // [rsp+8h] [rbp-58h]
  __int64 *v38; // [rsp+8h] [rbp-58h]
  int v39; // [rsp+10h] [rbp-50h]
  char *v40; // [rsp+10h] [rbp-50h]
  __int64 *v41; // [rsp+10h] [rbp-50h]
  __int64 *v42; // [rsp+10h] [rbp-50h]
  __int64 *v43; // [rsp+10h] [rbp-50h]
  __int64 *v44; // [rsp+18h] [rbp-48h]
  int v45; // [rsp+18h] [rbp-48h]
  __int64 *v46; // [rsp+18h] [rbp-48h]
  __int64 *v47; // [rsp+18h] [rbp-48h]
  __int64 *v48; // [rsp+20h] [rbp-40h]

  if ( !a4 )
    return;
  v8 = a5;
  if ( !a5 )
    return;
  v9 = a1;
  v10 = a4;
  v11 = a2;
  if ( a5 + a4 == 2 )
  {
    v47 = a1;
    v20 = a2;
LABEL_16:
    v42 = v20;
    v26 = *v20;
    v27 = *(__int64 **)(a7 + 64);
    v28 = *v47;
    if ( v27 )
    {
      v29 = sub_2E39EA0(v27, *v20);
      v30 = *(_QWORD *)(a7 + 64);
      v31 = v42;
      v32 = v29;
      if ( v30 )
      {
        v33 = sub_2E39EA0(*(__int64 **)(a7 + 64), v28);
        v30 = *(_QWORD *)(a7 + 64);
        v31 = v42;
        v34 = v33;
      }
      else
      {
        v34 = 0;
      }
      v43 = v31;
      v35 = sub_2EE68A0(*a8, *(_QWORD *)(a7 + 56), v30, 2);
      v22 = v43;
      if ( !v35 && v34 | v32 )
      {
        if ( v32 >= v34 )
          return;
LABEL_13:
        v25 = *v47;
        *v47 = *v22;
        *v22 = v25;
        return;
      }
    }
    else
    {
      sub_2EE68A0(*a8, *(_QWORD *)(a7 + 56), 0, 2);
      v22 = v42;
    }
    v48 = v22;
    v23 = sub_2E5E7B0(*(_QWORD *)(a7 + 48), v26);
    v24 = sub_2E5E7B0(*(_QWORD *)(a7 + 48), v28);
    v22 = v48;
    if ( v23 >= v24 )
      return;
    goto LABEL_13;
  }
  if ( a5 >= a4 )
    goto LABEL_10;
LABEL_5:
  v39 = (int)v9;
  v44 = v11;
  v12 = v10 / 2;
  v13 = &v9[v10 / 2];
  v14 = sub_2ED2B10(v11, a3, v13, (__int64 *)a7, a8);
  v15 = (char *)v44;
  v16 = v39;
  v17 = (char *)v14;
  v18 = v14 - v44;
  while ( 1 )
  {
    v45 = v16;
    v37 = (__int64 *)v17;
    v8 -= v18;
    v40 = sub_2ED2490((char *)v13, v15, v17);
    sub_2ED2D90(v45, (_DWORD)v13, (_DWORD)v40, v12, v18, v19, a7, (__int64)a8);
    v10 -= v12;
    if ( !v10 )
      break;
    v20 = v37;
    if ( !v8 )
      break;
    if ( v8 + v10 == 2 )
    {
      v47 = (__int64 *)v40;
      goto LABEL_16;
    }
    v11 = v37;
    v9 = (__int64 *)v40;
    if ( v8 < v10 )
      goto LABEL_5;
LABEL_10:
    v41 = v11;
    v46 = v9;
    v18 = v8 / 2;
    v38 = &v11[v8 / 2];
    v21 = sub_2ED2C50(v9, (__int64)v11, v38, (__int64 *)a7, a8);
    v16 = (int)v46;
    v17 = (char *)v38;
    v15 = (char *)v41;
    v13 = v21;
    v12 = v21 - v46;
  }
}
