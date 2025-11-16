// Function: sub_2814D30
// Address: 0x2814d30
//
_QWORD *__fastcall sub_2814D30(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rbx
  char v4; // al
  __int64 v5; // rdx
  __int64 v6; // rdi
  unsigned int v7; // eax
  unsigned int v8; // ecx
  __int64 v9; // rsi
  __int64 v10; // rdi
  unsigned int v11; // eax
  _QWORD *v12; // rax
  char v13; // dl
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r15
  _QWORD *v19; // r12
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r13
  char v25; // r15
  _QWORD *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rbx
  char v32; // al
  __int64 v33; // rdx
  __int64 v34; // rdi
  unsigned int v35; // eax
  unsigned int v36; // ecx
  __int64 v37; // rsi
  __int64 v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // r15
  char v44; // al
  __int64 v45; // rdx
  __int64 v46; // rsi
  unsigned int v47; // eax
  unsigned int v48; // edi
  __int64 v49; // rcx
  __int64 v50; // rsi
  unsigned int v51; // eax
  __int64 v52; // [rsp+0h] [rbp-50h]
  __int64 v53; // [rsp+0h] [rbp-50h]
  _QWORD *v54; // [rsp+8h] [rbp-48h]
  char v56; // [rsp+18h] [rbp-38h]
  __int64 v57; // [rsp+18h] [rbp-38h]
  char v58; // [rsp+18h] [rbp-38h]
  __int64 v59; // [rsp+18h] [rbp-38h]
  char v60; // [rsp+18h] [rbp-38h]

  v3 = (_QWORD *)a1[2];
  v54 = a1 + 1;
  if ( !v3 )
  {
    v3 = a1 + 1;
    goto LABEL_23;
  }
  while ( 1 )
  {
    v14 = *(_QWORD *)(a2 + 344);
    v15 = v3[4];
    v16 = v3[47];
    v17 = *(_QWORD *)(a2 + 368);
    if ( v14 )
      v18 = *(_QWORD *)(v14 + 40);
    else
      v18 = *(_QWORD *)a2;
    if ( v16 )
      v15 = *(_QWORD *)(v16 + 40);
    if ( (unsigned __int8)sub_B19720(*(_QWORD *)(a2 + 368), v15, v18) )
      break;
    if ( !(unsigned __int8)sub_B19720(v17, v18, v15) )
    {
      v56 = sub_29BD9C0(v18, v15, v17, *(_QWORD *)(a2 + 376));
      v4 = sub_29BD9C0(v15, v18, v17, *(_QWORD *)(a2 + 376));
      if ( v56 )
      {
        if ( !v4 )
          break;
        v5 = *(_QWORD *)(a2 + 376);
        if ( v18 )
        {
          v6 = (unsigned int)(*(_DWORD *)(v18 + 44) + 1);
          v7 = *(_DWORD *)(v18 + 44) + 1;
        }
        else
        {
          v6 = 0;
          v7 = 0;
        }
        v8 = *(_DWORD *)(v5 + 56);
        v9 = 0;
        if ( v7 < v8 )
          v9 = *(_QWORD *)(*(_QWORD *)(v5 + 48) + 8 * v6);
        if ( v15 )
        {
          v10 = (unsigned int)(*(_DWORD *)(v15 + 44) + 1);
          v11 = *(_DWORD *)(v15 + 44) + 1;
        }
        else
        {
          v10 = 0;
          v11 = 0;
        }
        if ( v8 <= v11 )
          BUG();
        if ( *(_DWORD *)(v9 + 16) <= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 48) + 8 * v10) + 16LL) )
          break;
      }
      else if ( !v4 )
      {
        goto LABEL_84;
      }
    }
    v12 = (_QWORD *)v3[2];
    v13 = 1;
    if ( !v12 )
      goto LABEL_22;
LABEL_15:
    v3 = v12;
  }
  v12 = (_QWORD *)v3[3];
  v13 = 0;
  if ( v12 )
    goto LABEL_15;
LABEL_22:
  v19 = v3;
  if ( !v13 )
    goto LABEL_25;
LABEL_23:
  if ( (_QWORD *)a1[3] == v3 )
  {
    v19 = v3;
    goto LABEL_34;
  }
  v19 = v3;
  v3 = (_QWORD *)sub_220EF80((__int64)v3);
LABEL_25:
  v20 = v3[47];
  v21 = *(_QWORD *)a2;
  v22 = *(_QWORD *)(a2 + 344);
  if ( v20 )
    v23 = *(_QWORD *)(v20 + 40);
  else
    v23 = v3[4];
  if ( v22 )
    v21 = *(_QWORD *)(v22 + 40);
  v57 = v3[50];
  if ( (unsigned __int8)sub_B19720(v57, v21, v23) )
    return v3;
  if ( !(unsigned __int8)sub_B19720(v57, v23, v21) )
  {
    v52 = v57;
    v58 = sub_29BD9C0(v23, v21, v57, v3[51]);
    v32 = sub_29BD9C0(v21, v23, v52, v3[51]);
    if ( v58 )
    {
      if ( !v32 )
        return v3;
      v33 = v3[51];
      if ( v23 )
      {
        v34 = (unsigned int)(*(_DWORD *)(v23 + 44) + 1);
        v35 = *(_DWORD *)(v23 + 44) + 1;
      }
      else
      {
        v34 = 0;
        v35 = 0;
      }
      v36 = *(_DWORD *)(v33 + 56);
      v37 = 0;
      if ( v35 < v36 )
        v37 = *(_QWORD *)(*(_QWORD *)(v33 + 48) + 8 * v34);
      if ( v21 )
      {
        v38 = (unsigned int)(*(_DWORD *)(v21 + 44) + 1);
        v39 = *(_DWORD *)(v21 + 44) + 1;
      }
      else
      {
        v38 = 0;
        v39 = 0;
      }
      if ( v36 <= v39 )
        BUG();
      if ( *(_DWORD *)(v37 + 16) <= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v33 + 48) + 8 * v38) + 16LL) )
        return v3;
      goto LABEL_33;
    }
    if ( v32 )
      goto LABEL_33;
    goto LABEL_84;
  }
LABEL_33:
  v3 = 0;
  if ( !v19 )
    return v3;
LABEL_34:
  if ( v54 != v19 )
  {
    v40 = *(_QWORD *)(a2 + 344);
    v41 = v19[4];
    v42 = v19[47];
    if ( v40 )
      v43 = *(_QWORD *)(v40 + 40);
    else
      v43 = *(_QWORD *)a2;
    if ( v42 )
      v41 = *(_QWORD *)(v42 + 40);
    v59 = *(_QWORD *)(a2 + 368);
    if ( (unsigned __int8)sub_B19720(v59, v41, v43) )
    {
LABEL_63:
      v25 = 0;
      goto LABEL_36;
    }
    if ( (unsigned __int8)sub_B19720(v59, v43, v41) )
      goto LABEL_35;
    v53 = v59;
    v60 = sub_29BD9C0(v43, v41, v59, *(_QWORD *)(a2 + 376));
    v44 = sub_29BD9C0(v41, v43, v53, *(_QWORD *)(a2 + 376));
    if ( v60 )
    {
      if ( v44 )
      {
        v45 = *(_QWORD *)(a2 + 376);
        if ( v43 )
        {
          v46 = (unsigned int)(*(_DWORD *)(v43 + 44) + 1);
          v47 = *(_DWORD *)(v43 + 44) + 1;
        }
        else
        {
          v46 = 0;
          v47 = 0;
        }
        v48 = *(_DWORD *)(v45 + 56);
        v49 = 0;
        if ( v47 < v48 )
          v49 = *(_QWORD *)(*(_QWORD *)(v45 + 48) + 8 * v46);
        if ( v41 )
        {
          v50 = (unsigned int)(*(_DWORD *)(v41 + 44) + 1);
          v51 = *(_DWORD *)(v41 + 44) + 1;
        }
        else
        {
          v50 = 0;
          v51 = 0;
        }
        if ( v48 <= v51 )
          BUG();
        v25 = *(_DWORD *)(v49 + 16) > *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v45 + 48) + 8 * v50) + 16LL);
        goto LABEL_36;
      }
      goto LABEL_63;
    }
    if ( v44 )
      goto LABEL_35;
LABEL_84:
    BUG();
  }
LABEL_35:
  v25 = 1;
LABEL_36:
  v26 = (_QWORD *)sub_22077B0(0x1A8u);
  v30 = *(unsigned int *)(a2 + 56);
  v31 = (__int64)v26;
  v26[4] = *(_QWORD *)a2;
  v26[5] = *(_QWORD *)(a2 + 8);
  v26[6] = *(_QWORD *)(a2 + 16);
  v26[7] = *(_QWORD *)(a2 + 24);
  v26[8] = *(_QWORD *)(a2 + 32);
  v26[9] = *(_QWORD *)(a2 + 40);
  v26[10] = v26 + 12;
  v26[11] = 0x1000000000LL;
  if ( (_DWORD)v30 )
    sub_2813E60((__int64)(v26 + 10), a2 + 48, v30, v27, v28, v29);
  *(_QWORD *)(v31 + 224) = v31 + 240;
  *(_QWORD *)(v31 + 232) = 0x1000000000LL;
  if ( *(_DWORD *)(a2 + 200) )
    sub_2813E60(v31 + 224, a2 + 192, v30, v27, v28, v29);
  *(_BYTE *)(v31 + 368) = *(_BYTE *)(a2 + 336);
  *(_QWORD *)(v31 + 376) = *(_QWORD *)(a2 + 344);
  *(_QWORD *)(v31 + 384) = *(_QWORD *)(a2 + 352);
  *(_WORD *)(v31 + 392) = *(_WORD *)(a2 + 360);
  *(_QWORD *)(v31 + 400) = *(_QWORD *)(a2 + 368);
  *(_QWORD *)(v31 + 408) = *(_QWORD *)(a2 + 376);
  *(_QWORD *)(v31 + 416) = *(_QWORD *)(a2 + 384);
  sub_220F040(v25, v31, v19, v54);
  ++a1[5];
  return (_QWORD *)v31;
}
