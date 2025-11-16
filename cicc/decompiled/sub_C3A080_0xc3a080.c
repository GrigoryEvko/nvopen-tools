// Function: sub_C3A080
// Address: 0xc3a080
//
__int64 __fastcall sub_C3A080(__int64 *a1, __int64 a2, unsigned int a3, int a4, char a5)
{
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned int v7; // ebx
  int v8; // edx
  unsigned int v9; // ebx
  unsigned int v10; // ebx
  __int64 v11; // rcx
  unsigned int v12; // ebx
  _QWORD *v13; // r9
  __int64 *v14; // r10
  int v15; // r12d
  __int64 v16; // r11
  unsigned int v17; // r13d
  unsigned int v18; // r14d
  _QWORD *v19; // r15
  __int64 *v20; // r15
  __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  unsigned int v24; // r12d
  int v25; // ebx
  int v26; // ebx
  bool v27; // dl
  _BOOL4 v28; // eax
  int v29; // ebx
  unsigned __int64 v30; // rbx
  unsigned __int64 *v31; // rdx
  unsigned int v32; // r8d
  unsigned __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rax
  unsigned int v37; // ebx
  __int64 v38; // r12
  unsigned int v39; // eax
  int v40; // edx
  unsigned int v41; // r12d
  __int64 *v43; // rcx
  int v44; // eax
  int v45; // edx
  unsigned __int64 *v46; // rax
  unsigned __int64 *v47; // rax
  int v48; // eax
  __int64 v50; // [rsp+20h] [rbp-4BC0h]
  _QWORD *v54; // [rsp+38h] [rbp-4BA8h]
  unsigned int v55; // [rsp+38h] [rbp-4BA8h]
  __int64 *v56; // [rsp+48h] [rbp-4B98h]
  __int64 *v57; // [rsp+48h] [rbp-4B98h]
  unsigned int v58; // [rsp+48h] [rbp-4B98h]
  __int64 *v59; // [rsp+50h] [rbp-4B90h]
  int v60; // [rsp+50h] [rbp-4B90h]
  unsigned int v61; // [rsp+50h] [rbp-4B90h]
  bool v62; // [rsp+5Bh] [rbp-4B85h]
  int v63; // [rsp+5Ch] [rbp-4B84h]
  unsigned int v64; // [rsp+5Ch] [rbp-4B84h]
  unsigned __int64 v65; // [rsp+60h] [rbp-4B80h] BYREF
  __int64 v66; // [rsp+68h] [rbp-4B78h]
  __int64 v67; // [rsp+70h] [rbp-4B70h]
  __int16 v68; // [rsp+78h] [rbp-4B68h]
  _QWORD v69[600]; // [rsp+80h] [rbp-4B60h] BYREF
  __int64 v70[2]; // [rsp+1340h] [rbp-38A0h] BYREF
  int v71; // [rsp+1350h] [rbp-3890h]
  _QWORD v72[1212]; // [rsp+2600h] [rbp-25E0h] BYREF

  v5 = (__int64)a1;
  v68 = 257;
  v65 = 0xFFFF800100007FFFLL;
  v6 = *a1;
  v66 = 0;
  v67 = 0;
  v62 = a5 == 4 || a5 == 1;
  v7 = *(_DWORD *)(v6 + 8) + 74;
  v8 = 1;
  v72[0] = 390625;
  v9 = v7 >> 6;
  if ( v9 )
    v8 = v9;
  v63 = v8;
  v10 = abs32(a4);
  v11 = v10 & 7;
  v12 = v10 >> 3;
  v69[0] = qword_3F65540[v11];
  if ( !v12 )
  {
    v55 = 1;
    goto LABEL_20;
  }
  v13 = v72;
  v14 = v70;
  v15 = 0;
  v16 = 1;
  v17 = 1;
  v18 = 1;
  v59 = v69;
  while ( 1 )
  {
    if ( (v12 & 1) == 0 )
      goto LABEL_5;
    v20 = v59;
    v21 = v18;
    v50 = v16;
    v18 += v17;
    v54 = v13;
    v57 = v14;
    sub_C47530(v14, v59, v13, v21, v17);
    v13 = v54;
    v16 = v50;
    if ( !v57[v18 - 1] )
      break;
    v43 = v59;
    v59 = v57;
    v14 = v43;
LABEL_5:
    ++v15;
    v12 >>= 1;
    v19 = &v13[v16];
    if ( !v12 )
      goto LABEL_13;
LABEL_6:
    if ( v15 )
    {
      v56 = v14;
      sub_C47530(v19, v13, v13, v17, v17);
      v14 = v56;
      v17 *= 2;
      v16 = v17;
      if ( !v19[v17 - 1] )
      {
        --v17;
        v16 = (unsigned int)(v16 - 1);
      }
    }
    v13 = v19;
  }
  ++v15;
  v12 >>= 1;
  v59 = v57;
  --v18;
  v14 = v20;
  v19 = &v54[v50];
  if ( v12 )
    goto LABEL_6;
LABEL_13:
  v55 = v18;
  v5 = (__int64)a1;
  if ( v59 != v69 )
    sub_C45D30(v69, v59, v18);
  v6 = *a1;
  while ( 1 )
  {
LABEL_20:
    LODWORD(v66) = (v63 << 6) - 1;
    v24 = v66 - *(_DWORD *)(v6 + 8);
    sub_C373C0(v70, (__int64)&v65);
    sub_C37310((__int64)v70, (*(_BYTE *)(v5 + 20) & 8) != 0);
    sub_C37380(v72, (__int64)&v65);
    v60 = sub_C367B0((__int64)v70, a2, a3, 1);
    v25 = sub_C367B0((__int64)v72, (__int64)v69, v55, 1);
    v71 += a4;
    if ( a4 < 0 )
    {
      v44 = sub_C33FE0(v70, (__int64)v72);
      v58 = v24;
      v45 = *(_DWORD *)(*(_QWORD *)v5 + 4LL);
      if ( v71 < v45 )
      {
        v58 = v45 - v71 + v24;
        v24 = v58;
        if ( (unsigned int)v66 <= v58 )
          v24 = v66;
      }
      v26 = v44 | v25;
      if ( v26 )
      {
        v27 = v44 != 0;
        v29 = 2 - ((v60 == 0) - 1);
        goto LABEL_36;
      }
      v28 = 0;
      v27 = 0;
    }
    else
    {
      v58 = v24;
      v26 = v25 != 0;
      v27 = (unsigned int)sub_C3A020(v70, (__int64)v72) != 0;
      v28 = v27;
    }
    v29 = v26 - ((v60 == 0) - 1);
    if ( !v29 )
    {
      v30 = (unsigned int)(2 * v28);
      goto LABEL_24;
    }
LABEL_36:
    v30 = (unsigned int)v27 + 2 * v29;
LABEL_24:
    v31 = (unsigned __int64 *)sub_C33900((__int64)v70);
    v32 = (v24 - 1) >> 6;
    v33 = v31[v32] & (0xFFFFFFFFFFFFFFFFLL >> (63 - ((v24 - 1) & 0x3F)));
    v34 = 1LL << ((v24 - 1) & 0x3F);
    if ( !v62 )
      v34 = 0;
    if ( v24 - 1 <= 0x3F )
    {
      v22 = v33 - v34;
      v23 = v34 - v33;
      if ( v22 <= v23 )
        v23 = v22;
      goto LABEL_18;
    }
    if ( v33 != v34 )
      break;
    v46 = &v31[v32 - 1];
    while ( v31 != v46 )
    {
      --v46;
      if ( v46[1] )
        goto LABEL_29;
    }
    v23 = *v31;
LABEL_18:
    if ( v30 <= 2 * v23 )
      goto LABEL_29;
    sub_C338F0((__int64)v72);
    sub_C338F0((__int64)v70);
    v6 = *(_QWORD *)v5;
    v63 *= 2;
  }
  if ( v33 == v34 - 1 )
  {
    v47 = &v31[v32 - 1];
    while ( v31 != v47 )
    {
      --v47;
      if ( v47[1] != -1 )
        goto LABEL_29;
    }
    v23 = -(__int64)*v31;
    goto LABEL_18;
  }
LABEL_29:
  v61 = v66 - v24;
  v35 = sub_C33900((__int64)v70);
  v64 = sub_C337D0(v5);
  v36 = sub_C33900(v5);
  sub_C49830(v36, v64, v35, v61, v24);
  *(_DWORD *)(v5 + 16) = *(_DWORD *)(*(_QWORD *)v5 + 8LL) + v71 - v66 + v24;
  v37 = sub_C337D0((__int64)v70);
  v38 = sub_C33900((__int64)v70);
  v39 = sub_C45DF0(v38, v37);
  v40 = 0;
  if ( v39 < v58 )
  {
    if ( v39 + 1 == v58 )
    {
      v40 = 2;
    }
    else if ( v37 << 6 < v58 || (v48 = sub_C45D90(v38, v58 - 1), v40 = 3, !v48) )
    {
      v40 = 1;
    }
  }
  v41 = sub_C36450(v5, a5, v40);
  sub_C338F0((__int64)v72);
  sub_C338F0((__int64)v70);
  return v41;
}
