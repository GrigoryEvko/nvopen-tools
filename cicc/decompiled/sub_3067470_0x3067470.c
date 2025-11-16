// Function: sub_3067470
// Address: 0x3067470
//
_BOOL8 __fastcall sub_3067470(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v8; // rdx
  _BOOL8 v9; // r13
  __int64 v11; // r12
  signed __int64 v12; // rbx
  _BYTE *v13; // r14
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // r13
  __int64 v16; // rdx
  _BYTE *v17; // rax
  _QWORD *v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  _BYTE *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r10
  __int64 v27; // rax
  unsigned __int64 v28; // rsi
  __int64 v29; // rdx
  unsigned __int64 v30; // r10
  unsigned int v31; // eax
  __int64 v32; // rbx
  char v33; // al
  __int64 v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // rsi
  unsigned __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rcx
  __int64 v41; // rax
  unsigned int v42; // ebx
  unsigned int v43; // ebx
  __int64 v44; // rcx
  __int64 v45; // rdi
  __int64 v46; // rsi
  char v47; // bl
  unsigned __int64 v49; // [rsp+0h] [rbp-E0h]
  unsigned __int8 *v50; // [rsp+8h] [rbp-D8h]
  char v52; // [rsp+1Bh] [rbp-C5h]
  unsigned int v53; // [rsp+1Ch] [rbp-C4h]
  unsigned __int64 v54; // [rsp+20h] [rbp-C0h]
  __int64 v55; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v56; // [rsp+30h] [rbp-B0h]
  __int64 v57; // [rsp+38h] [rbp-A8h]
  char v58; // [rsp+38h] [rbp-A8h]
  __int64 v59; // [rsp+38h] [rbp-A8h]
  __int64 v60; // [rsp+38h] [rbp-A8h]
  __int64 v61; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v63; // [rsp+50h] [rbp-90h]
  __int64 v64; // [rsp+58h] [rbp-88h]
  char *v65; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v66; // [rsp+68h] [rbp-78h]
  unsigned __int64 v67; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v68; // [rsp+78h] [rbp-68h]
  unsigned __int64 v69; // [rsp+80h] [rbp-60h] BYREF
  __int64 v70; // [rsp+88h] [rbp-58h]
  char v71; // [rsp+90h] [rbp-50h]
  unsigned __int64 v72; // [rsp+98h] [rbp-48h]
  __int64 v73; // [rsp+A0h] [rbp-40h]

  v52 = 0;
  v50 = sub_BD3990((unsigned __int8 *)a3, a2);
  if ( *v50 > 3u )
  {
    v52 = 1;
    v50 = 0;
  }
  v53 = sub_AE43A0(*a1, *(_QWORD *)(a3 + 8));
  v66 = v53;
  if ( v53 > 0x40 )
    sub_C43690((__int64)&v65, 0, 0);
  else
    v65 = 0;
  if ( !a5 )
  {
    v9 = v50 != 0;
    goto LABEL_7;
  }
  v61 = a4 + 8 * a5;
  if ( a4 != v61 )
  {
    v11 = a4 + 8;
    v54 = 0;
    v12 = a2 & 0xFFFFFFFFFFFFFFF9LL | 4;
    while ( 1 )
    {
      v64 = v11;
      v13 = *(_BYTE **)(v11 - 8);
      v14 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      v15 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v12 )
      {
        v34 = sub_BCBAE0(v14, *(unsigned __int8 **)(v11 - 8), v8);
        v13 = *(_BYTE **)(v11 - 8);
        v63 = v34;
        if ( *v13 == 17 )
          goto LABEL_36;
        v35 = (_BYTE *)sub_9B7920(v13);
        v13 = v35;
        if ( !v35 )
          goto LABEL_36;
        if ( *v35 != 17 )
          v13 = 0;
        if ( sub_BCEA30(v63) )
          goto LABEL_68;
LABEL_37:
        v26 = *a1;
        goto LABEL_38;
      }
      v16 = (v12 >> 1) & 3;
      if ( v16 == 2 )
        break;
      if ( v16 == 1 )
      {
        if ( v14 )
        {
          v63 = *(_QWORD *)(v14 + 24);
          if ( *v13 == 17 )
            goto LABEL_54;
LABEL_19:
          v17 = (_BYTE *)sub_9B7920(v13);
          v16 = (v12 >> 1) & 3;
          v13 = v17;
          if ( !v17 )
            goto LABEL_23;
          if ( *v17 != 17 )
            v13 = 0;
LABEL_22:
          v16 = (v12 >> 1) & 3;
          goto LABEL_23;
        }
LABEL_32:
        v23 = sub_BCBAE0(0, v13, v16);
        v13 = *(_BYTE **)(v11 - 8);
        v16 = (v12 >> 1) & 3;
        v63 = v23;
        if ( *v13 == 17 )
          goto LABEL_54;
LABEL_33:
        v24 = (_BYTE *)sub_9B7920(v13);
        v13 = v24;
        if ( v24 && *v24 != 17 )
          v13 = 0;
        goto LABEL_22;
      }
      v38 = sub_BCBAE0(v12 & 0xFFFFFFFFFFFFFFF8LL, *(unsigned __int8 **)(v11 - 8), v16);
      v13 = *(_BYTE **)(v11 - 8);
      v16 = (v12 >> 1) & 3;
      v63 = v38;
      if ( *v13 != 17 )
        goto LABEL_33;
LABEL_23:
      if ( !v16 )
      {
        if ( v14 )
        {
          v18 = (_QWORD *)*((_QWORD *)v13 + 3);
          if ( *((_DWORD *)v13 + 8) > 0x40u )
            v18 = (_QWORD *)*v18;
          v19 = 16LL * (unsigned int)v18 + sub_AE4AC0(*a1, v14) + 24;
          v20 = *(_QWORD *)v19;
          LOBYTE(v19) = *(_BYTE *)(v19 + 8);
          v69 = v20;
          LOBYTE(v70) = v19;
          v21 = sub_CA1930(&v69);
          sub_C46A40((__int64)&v65, v21);
          goto LABEL_28;
        }
LABEL_36:
        if ( sub_BCEA30(v63) )
          goto LABEL_68;
        goto LABEL_37;
      }
LABEL_54:
      v59 = v16;
      v33 = sub_BCEA30(v63);
      v25 = v59;
      if ( v33 )
        goto LABEL_68;
      v26 = *a1;
      if ( v59 == 2 )
      {
        v28 = v12 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v14 )
        {
LABEL_38:
          v57 = v26;
          v27 = sub_BCBAE0(v12 & 0xFFFFFFFFFFFFFFF8LL, *(unsigned __int8 **)(v11 - 8), v25);
          v26 = v57;
          v28 = v27;
        }
        v55 = v26;
        v58 = sub_AE5020(v26, v28);
        v69 = sub_9208B0(v55, v28);
        v70 = v29;
        v22 = 1LL << v58;
        v30 = (((v69 + 7) >> 3) + (1LL << v58) - 1) >> v58 << v58;
        goto LABEL_40;
      }
      if ( v59 != 1 )
        goto LABEL_38;
      if ( v14 )
      {
        v36 = *(_QWORD *)(v14 + 24);
      }
      else
      {
        v60 = *a1;
        v39 = sub_BCBAE0(0, *(unsigned __int8 **)(v11 - 8), 1);
        v26 = v60;
        v36 = v39;
      }
      v69 = sub_9208B0(v26, v36);
      v70 = v22;
      v30 = (v69 + 7) >> 3;
LABEL_40:
      if ( v13 )
      {
        v56 = v30;
        sub_C44B10((__int64)&v67, (char **)v13 + 3, v53);
        sub_C47170((__int64)&v67, v56);
        v31 = v68;
        v68 = 0;
        LODWORD(v70) = v31;
        v69 = v67;
        sub_C45EE0((__int64)&v65, (__int64 *)&v69);
        if ( (unsigned int)v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        if ( v68 > 0x40 && v67 )
          j_j___libc_free_0_0(v67);
      }
      else
      {
        if ( v54 )
          goto LABEL_68;
        v54 = v30;
      }
      if ( v12 )
      {
        v32 = (v12 >> 1) & 3;
        if ( v32 == 2 )
        {
          if ( v14 )
            goto LABEL_29;
        }
        else if ( v32 == 1 && v14 )
        {
          v15 = *(_QWORD *)(v14 + 24);
          goto LABEL_29;
        }
      }
LABEL_28:
      v15 = sub_BCBAE0(v14, *(unsigned __int8 **)(v11 - 8), v22);
LABEL_29:
      v8 = *(unsigned __int8 *)(v15 + 8);
      if ( (_BYTE)v8 == 16 )
      {
        v12 = *(_QWORD *)(v15 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
      }
      else if ( (unsigned int)(unsigned __int8)v8 - 17 > 1 )
      {
        v37 = v15 & 0xFFFFFFFFFFFFFFF9LL;
        v12 = 0;
        if ( (_BYTE)v8 == 15 )
          v12 = v37;
      }
      else
      {
        v12 = v15 & 0xFFFFFFFFFFFFFFF9LL | 2;
      }
      v11 += 8;
      if ( v64 == v61 )
        goto LABEL_82;
    }
    v63 = v12 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v14 )
    {
      if ( *v13 == 17 )
        goto LABEL_54;
      goto LABEL_19;
    }
    goto LABEL_32;
  }
  v63 = 0;
  v54 = 0;
LABEL_82:
  v40 = v63;
  if ( a6 )
    v40 = a6;
  v41 = *(_QWORD *)(a3 + 8);
  v49 = v40;
  if ( (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17 <= 1 )
    v41 = **(_QWORD **)(v41 + 16);
  v42 = *(_DWORD *)(v41 + 8);
  sub_C44B10((__int64)&v67, &v65, 0x40u);
  v43 = v42 >> 8;
  if ( v68 <= 0x40 )
  {
    v44 = 0;
    if ( v68 )
      v44 = (__int64)(v67 << (64 - (unsigned __int8)v68)) >> (64 - (unsigned __int8)v68);
  }
  else
  {
    v44 = *(_QWORD *)v67;
  }
  v70 = v44;
  v73 = 0;
  v69 = (unsigned __int64)v50;
  v45 = a1[2];
  v46 = *a1;
  v71 = v52;
  v72 = v54;
  v47 = (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64 *, unsigned __int64, _QWORD, _QWORD))(*(_QWORD *)v45 + 1288LL))(
          v45,
          v46,
          &v69,
          v49,
          v43,
          0);
  if ( v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  if ( v47 )
    v9 = 0;
  else
LABEL_68:
    v9 = 1;
LABEL_7:
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0((unsigned __int64)v65);
  return v9;
}
