// Function: sub_395A530
// Address: 0x395a530
//
__int64 __fastcall sub_395A530(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  int v9; // r12d
  bool v10; // bl
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v14; // r12
  __int64 v15; // r14
  __int64 v16; // rsi
  __int64 v17; // r11
  bool v18; // zf
  __int64 *v19; // rax
  __int64 v20; // r11
  __int64 v21; // rdx
  __int64 *v22; // rcx
  unsigned int v23; // eax
  const void *v24; // rsi
  __int64 v25; // r14
  __int64 v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r10
  int v32; // eax
  unsigned int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r10
  int v37; // eax
  int v38; // eax
  int v39; // eax
  __int64 *v40; // r14
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // r11
  int v44; // eax
  int v45; // eax
  int v46; // r12d
  bool v47; // bl
  __int64 v48; // rax
  int v49; // eax
  __int64 v50; // [rsp+0h] [rbp-70h]
  __int64 v51; // [rsp+0h] [rbp-70h]
  __int64 v53; // [rsp+18h] [rbp-58h]
  __int64 v54; // [rsp+18h] [rbp-58h]
  __int64 v58; // [rsp+30h] [rbp-40h]
  __int64 *v59; // [rsp+38h] [rbp-38h]
  __int64 v60; // [rsp+38h] [rbp-38h]

  v7 = *(unsigned int *)(a4 + 8);
  v8 = *(_QWORD *)(a5 + 104);
  if ( !(_DWORD)v7 )
  {
    v9 = *(_DWORD *)(a5 + 112);
    v10 = *(_QWORD *)a5 == 0x200000002LL;
    v11 = sub_1644900(a3, 0x20u);
    v12 = sub_159C470(v11, 0, 0);
    *(_DWORD *)(a1 + 8) = v10 + 1;
    *(_QWORD *)a1 = v12;
    *(_DWORD *)(a1 + 12) = v9;
    *(_QWORD *)(a1 + 16) = v8;
    return a1;
  }
  v14 = *(__int64 **)a4;
  v15 = *(_QWORD *)a4 + 24 * v7;
  v59 = (__int64 *)v15;
  while ( 1 )
  {
    v20 = *v14;
    if ( *(_BYTE *)(*v14 + 16) <= 0x17u )
    {
      if ( *(_QWORD *)a5 != 0x200000002LL )
        goto LABEL_18;
      v54 = *v14;
      v34 = sub_15F2050(*(_QWORD *)(a5 + 104));
      v35 = sub_1632FA0(v34);
      v20 = v54;
      v36 = v35;
      v37 = *(unsigned __int8 *)(v54 + 16);
      if ( (unsigned __int8)v37 <= 0x17u )
      {
        v51 = v36;
        if ( (unsigned int)sub_14C23D0(v54, v36, 0, 0, 0, 0) )
        {
          v49 = sub_3959780(v51, (__int64 *)v54);
          v20 = v54;
          if ( v49 == 2 )
          {
LABEL_18:
            if ( sub_395A240(a6, v20, 0x20u, 0) )
              goto LABEL_19;
            goto LABEL_15;
          }
        }
      }
      else
      {
        v38 = v37 - 24;
        if ( v38 == 37 || v38 == 24 )
          goto LABEL_18;
      }
      goto LABEL_15;
    }
    if ( !byte_5054A00 )
      goto LABEL_57;
    if ( v20 == v8 )
    {
      v17 = v8;
      if ( *(_QWORD *)a5 != 0x200000002LL )
        goto LABEL_12;
      goto LABEL_24;
    }
    if ( !sub_15CCEE0(a2, *v14, v8) )
    {
LABEL_57:
      v16 = v14[2];
      if ( !v16 || v16 != v8 && !sub_15CCEE0(a2, v16, v8) )
        break;
    }
    v17 = *v14;
    if ( *(_QWORD *)a5 != 0x200000002LL )
      goto LABEL_12;
LABEL_24:
    v53 = v17;
    v29 = sub_15F2050(*(_QWORD *)(a5 + 104));
    v30 = sub_1632FA0(v29);
    v17 = v53;
    v31 = v30;
    v32 = *(unsigned __int8 *)(v53 + 16);
    if ( (unsigned __int8)v32 <= 0x17u )
    {
      v50 = v31;
      if ( (unsigned int)sub_14C23D0(v53, v31, 0, 0, 0, 0) )
      {
        v39 = sub_3959780(v50, (__int64 *)v53);
        v17 = v53;
        if ( v39 == 2 )
        {
LABEL_12:
          v18 = !sub_395A240(a6, v17, 0x20u, 0);
          v19 = v59;
          if ( !v18 )
            v19 = v14;
          v59 = v19;
        }
      }
LABEL_15:
      v14 += 3;
      if ( (__int64 *)v15 == v14 )
        break;
    }
    else
    {
      v33 = v32 - 24;
      if ( v33 == 37 )
        goto LABEL_12;
      if ( v33 > 0x25 )
        goto LABEL_15;
      if ( v33 == 24 )
        goto LABEL_12;
      v14 += 3;
      if ( (__int64 *)v15 == v14 )
        break;
    }
  }
  v14 = v59;
LABEL_19:
  v21 = *(unsigned int *)(a4 + 8);
  v22 = *(__int64 **)a4;
  v23 = *(_DWORD *)(a4 + 8);
  if ( v14 != (__int64 *)v15 )
  {
    v24 = v14 + 3;
    v25 = *v14;
    v26 = v14[2];
    v27 = (__int64)&v22[3 * v21];
    v28 = v14[1];
    if ( (__int64 *)v27 != v14 + 3 )
      goto LABEL_21;
    goto LABEL_22;
  }
  v40 = &v22[3 * v21];
  if ( v40 == v22 )
  {
LABEL_48:
    v46 = *(_DWORD *)(a5 + 112);
    v47 = *(_QWORD *)a5 == 0x200000002LL;
    v48 = sub_1644900(a3, 0x20u);
    *(_QWORD *)a1 = sub_159C470(v48, 0, 0);
    *(_DWORD *)(a1 + 8) = v47 + 1;
    *(_DWORD *)(a1 + 12) = v46;
    *(_QWORD *)(a1 + 16) = v8;
    return a1;
  }
  v60 = v8;
  v14 = *(__int64 **)a4;
  while ( 2 )
  {
    v41 = *v14;
    if ( *(_QWORD *)a5 == 0x200000002LL )
    {
      v42 = sub_15F2050(*(_QWORD *)(a5 + 104));
      v43 = sub_1632FA0(v42);
      v44 = *(unsigned __int8 *)(v41 + 16);
      if ( (unsigned __int8)v44 <= 0x17u )
      {
        v58 = v43;
        if ( (unsigned int)sub_14C23D0(v41, v43, 0, 0, 0, 0) && (unsigned int)sub_3959780(v58, (__int64 *)v41) == 2 )
          break;
      }
      else
      {
        v45 = v44 - 24;
        if ( v45 == 37 || v45 == 24 )
          break;
      }
LABEL_46:
      v14 += 3;
      if ( v40 == v14 )
      {
        v8 = v60;
        goto LABEL_48;
      }
      continue;
    }
    break;
  }
  if ( !sub_395A240(a6, v41, 0x20u, 0) )
    goto LABEL_46;
  v24 = v14 + 3;
  v25 = *v14;
  v26 = v14[2];
  v28 = v14[1];
  v23 = *(_DWORD *)(a4 + 8);
  v27 = *(_QWORD *)a4 + 24LL * v23;
  if ( (__int64 *)v27 == v14 + 3 )
    goto LABEL_22;
LABEL_21:
  memmove(v14, v24, v27 - (_QWORD)v24);
  v23 = *(_DWORD *)(a4 + 8);
LABEL_22:
  *(_DWORD *)(a4 + 8) = v23 - 1;
  *(_QWORD *)a1 = v25;
  *(_QWORD *)(a1 + 8) = v28;
  *(_QWORD *)(a1 + 16) = v26;
  return a1;
}
