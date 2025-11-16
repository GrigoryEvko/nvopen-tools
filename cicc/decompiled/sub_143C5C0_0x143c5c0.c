// Function: sub_143C5C0
// Address: 0x143c5c0
//
__int64 __fastcall sub_143C5C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  bool v13; // cc
  __int64 v14; // r11
  unsigned __int8 v16; // al
  __int64 v17; // r13
  __int64 v18; // r15
  __int64 v19; // rax
  const char *v20; // rdx
  int v21; // edi
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rsi
  __int64 v25; // r12
  __int64 v26; // rax
  unsigned int v27; // eax
  __int64 v28; // r9
  __int64 i; // r13
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 **v34; // r13
  __int64 v35; // r14
  __int64 *v36; // r15
  __int64 v37; // rax
  unsigned int v38; // r8d
  __int64 v39; // r11
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r8
  __int64 v44; // r11
  __int64 v45; // r10
  __int64 **v46; // rcx
  __int64 **v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // r9
  _QWORD *v51; // rsi
  _QWORD *v52; // r13
  unsigned __int8 v53; // al
  __int64 v54; // rax
  _QWORD *v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // [rsp+0h] [rbp-140h]
  unsigned int v59; // [rsp+8h] [rbp-138h]
  __int64 v60; // [rsp+8h] [rbp-138h]
  __int64 v61; // [rsp+10h] [rbp-130h]
  unsigned int v62; // [rsp+10h] [rbp-130h]
  __int64 v63; // [rsp+20h] [rbp-120h]
  unsigned int v64; // [rsp+20h] [rbp-120h]
  __int64 v65; // [rsp+20h] [rbp-120h]
  __int64 v66; // [rsp+28h] [rbp-118h]
  __int64 v67; // [rsp+28h] [rbp-118h]
  __int64 v68; // [rsp+28h] [rbp-118h]
  __int64 v69; // [rsp+28h] [rbp-118h]
  __int64 v70; // [rsp+30h] [rbp-110h]
  __int64 v71; // [rsp+30h] [rbp-110h]
  __int64 v72; // [rsp+30h] [rbp-110h]
  __int64 v73; // [rsp+30h] [rbp-110h]
  __int64 v74; // [rsp+30h] [rbp-110h]
  __int64 v75; // [rsp+30h] [rbp-110h]
  __int64 v76; // [rsp+30h] [rbp-110h]
  __int64 v77; // [rsp+30h] [rbp-110h]
  __int64 v78; // [rsp+30h] [rbp-110h]
  __int64 v79; // [rsp+30h] [rbp-110h]
  __int64 v80; // [rsp+30h] [rbp-110h]
  __int64 v81; // [rsp+38h] [rbp-108h]
  _QWORD v82[2]; // [rsp+40h] [rbp-100h] BYREF
  _QWORD *v83; // [rsp+50h] [rbp-F0h] BYREF
  const char *v84; // [rsp+58h] [rbp-E8h]
  __int16 v85; // [rsp+60h] [rbp-E0h]
  __int64 v86[4]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD *v87; // [rsp+90h] [rbp-B0h]
  __int64 v88; // [rsp+98h] [rbp-A8h]
  _QWORD v89[4]; // [rsp+A0h] [rbp-A0h] BYREF
  _WORD *v90; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v91; // [rsp+C8h] [rbp-78h]
  _WORD v92[56]; // [rsp+D0h] [rbp-70h] BYREF

  v9 = a2;
  v11 = *(_QWORD *)(a1 + 24);
  v12 = *(_QWORD *)(a1 + 8);
  v86[0] = a2;
  v86[2] = 0;
  v13 = *(_BYTE *)(a2 + 16) <= 0x17u;
  v86[3] = v11;
  v87 = v89;
  v86[1] = v12;
  v88 = 0x400000000LL;
  if ( !v13 )
  {
    v89[0] = a2;
    LODWORD(v88) = 1;
  }
  v70 = a3;
  if ( !sub_143C480(v86, a3, a4, a5, 1) )
  {
    v14 = v86[0];
    goto LABEL_5;
  }
  v16 = *(_BYTE *)(a2 + 16);
  if ( v16 <= 0x17u )
    goto LABEL_19;
  if ( (unsigned int)v16 - 60 <= 0xC )
  {
    if ( (unsigned __int8)sub_14AF470(a2, 0, 0, 0) )
    {
      v17 = sub_143C5C0(a1, *(_QWORD *)(a2 - 24), v70, a4, a5);
      if ( v17 )
      {
        v18 = sub_157EBA0(a4);
        v19 = sub_1649960(a2);
        v84 = v20;
        v21 = *(unsigned __int8 *)(a2 + 16);
        v83 = (_QWORD *)v19;
        v92[0] = 773;
        v22 = *(_QWORD *)a2;
        v90 = &v83;
        v91 = (__int64)".phi.trans.insert";
        v23 = sub_15FDBD0((unsigned int)(v21 - 24), v17, v22, &v90, v18);
        v24 = *(_QWORD *)(a2 + 48);
        v14 = v23;
        v90 = (_WORD *)v24;
        if ( v24 )
        {
          v71 = v23;
          sub_1623A60(&v90, v24, 2);
          v14 = v71;
          v25 = v71 + 48;
          if ( (_WORD **)(v71 + 48) == &v90 )
          {
            if ( v90 )
            {
              sub_161E7C0(v71 + 48);
              v14 = v71;
            }
            goto LABEL_16;
          }
          if ( !*(_QWORD *)(v71 + 48) )
          {
LABEL_31:
            v32 = (unsigned __int64)v90;
            *(_QWORD *)(v14 + 48) = v90;
            if ( v32 )
            {
              v74 = v14;
              sub_1623210(&v90, v32, v25);
              v14 = v74;
            }
            goto LABEL_16;
          }
        }
        else
        {
          v25 = v23 + 48;
          if ( (_WORD **)(v23 + 48) == &v90 || !*(_QWORD *)(v23 + 48) )
          {
LABEL_16:
            v26 = *(unsigned int *)(a6 + 8);
            if ( (unsigned int)v26 >= *(_DWORD *)(a6 + 12) )
            {
              v78 = v14;
              sub_16CD150(a6, a6 + 16, 0, 8);
              v26 = *(unsigned int *)(a6 + 8);
              v14 = v78;
            }
            *(_QWORD *)(*(_QWORD *)a6 + 8 * v26) = v14;
            ++*(_DWORD *)(a6 + 8);
            goto LABEL_5;
          }
        }
        v73 = v14;
        sub_161E7C0(v25);
        v14 = v73;
        goto LABEL_31;
      }
    }
LABEL_19:
    v14 = 0;
    goto LABEL_5;
  }
  if ( v16 != 56 )
    goto LABEL_19;
  v90 = v92;
  v91 = 0x800000000LL;
  v66 = *(_QWORD *)(a2 + 40);
  v27 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v27 )
  {
    v28 = a6;
    v63 = v27 - 1;
    for ( i = 0; ; ++i )
    {
      v72 = v28;
      v30 = sub_143C5C0(a1, *(_QWORD *)(a2 + 24 * (i - v27)), v66, a4, a5);
      if ( !v30 )
        break;
      v31 = (unsigned int)v91;
      v28 = v72;
      if ( (unsigned int)v91 >= HIDWORD(v91) )
      {
        v60 = v30;
        sub_16CD150(&v90, v92, 0, 8);
        v31 = (unsigned int)v91;
        v30 = v60;
        v28 = v72;
      }
      *(_QWORD *)&v90[4 * v31] = v30;
      LODWORD(v91) = v91 + 1;
      if ( v63 == i )
      {
        v9 = a2;
        a6 = v28;
        goto LABEL_36;
      }
      v27 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
    v14 = 0;
    goto LABEL_54;
  }
LABEL_36:
  v67 = sub_157EBA0(a4);
  v82[0] = sub_1649960(v9);
  v85 = 773;
  v83 = v82;
  v84 = ".phi.trans.insert";
  v82[1] = v33;
  v34 = (__int64 **)(v90 + 4);
  v35 = (unsigned int)v91 - 1LL;
  v36 = *(__int64 **)v90;
  v75 = *(_QWORD *)(v9 + 56);
  if ( !v75 )
  {
    v56 = *v36;
    if ( *(_BYTE *)(*v36 + 8) == 16 )
      v56 = **(_QWORD **)(v56 + 16);
    v75 = *(_QWORD *)(v56 + 24);
  }
  v61 = (unsigned int)v91;
  v64 = v91;
  v37 = sub_1648A60(72, (unsigned int)v91);
  v38 = v64;
  v39 = v37;
  if ( v37 )
  {
    v40 = *v36;
    v65 = v39 - 24 * v61;
    if ( *(_BYTE *)(*v36 + 8) == 16 )
      v40 = **(_QWORD **)(v40 + 16);
    v58 = v39;
    v59 = v38;
    v62 = *(_DWORD *)(v40 + 8) >> 8;
    v41 = sub_15F9F50(v75, v34, v35);
    v42 = sub_1646BA0(v41, v62);
    v43 = v59;
    v44 = v58;
    v45 = v42;
    if ( *(_BYTE *)(*v36 + 8) == 16 )
    {
      v57 = sub_16463B0(v42, *(_QWORD *)(*v36 + 32));
      v43 = v59;
      v44 = v58;
      v45 = v57;
    }
    else
    {
      v46 = &v34[v35];
      if ( v34 != v46 )
      {
        v47 = v34;
        while ( 1 )
        {
          v48 = **v47;
          if ( *(_BYTE *)(v48 + 8) == 16 )
            break;
          if ( v46 == ++v47 )
            goto LABEL_46;
        }
        v49 = sub_16463B0(v45, *(_QWORD *)(v48 + 32));
        v44 = v58;
        v43 = v59;
        v45 = v49;
      }
    }
LABEL_46:
    v50 = v67;
    v68 = v44;
    sub_15F1EA0(v44, v45, 32, v65, v43, v50);
    *(_QWORD *)(v68 + 56) = v75;
    *(_QWORD *)(v68 + 64) = sub_15F9F50(v75, v34, v35);
    sub_15F9CE0(v68, v36, v34, v35, &v83);
    v39 = v68;
  }
  v51 = *(_QWORD **)(v9 + 48);
  v52 = (_QWORD *)(v39 + 48);
  v83 = v51;
  if ( v51 )
  {
    v69 = v39;
    sub_1623A60(&v83, v51, 2);
    v39 = v69;
    if ( v52 == &v83 )
    {
      if ( v83 )
      {
        sub_161E7C0(&v83);
        v39 = v69;
      }
      goto LABEL_51;
    }
    if ( !*(_QWORD *)(v69 + 48) )
    {
LABEL_60:
      v55 = v83;
      *(_QWORD *)(v39 + 48) = v83;
      if ( v55 )
      {
        v80 = v39;
        sub_1623210(&v83, v55, v52);
        v39 = v80;
      }
      goto LABEL_51;
    }
LABEL_59:
    v79 = v39;
    sub_161E7C0(v52);
    v39 = v79;
    goto LABEL_60;
  }
  if ( v52 != &v83 && *(_QWORD *)(v39 + 48) )
    goto LABEL_59;
LABEL_51:
  v76 = v39;
  v53 = sub_15FA300(v9);
  sub_15FA2E0(v76, v53);
  v54 = *(unsigned int *)(a6 + 8);
  v14 = v76;
  if ( (unsigned int)v54 >= *(_DWORD *)(a6 + 12) )
  {
    sub_16CD150(a6, a6 + 16, 0, 8);
    v54 = *(unsigned int *)(a6 + 8);
    v14 = v76;
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v54) = v14;
  ++*(_DWORD *)(a6 + 8);
LABEL_54:
  if ( v90 != v92 )
  {
    v77 = v14;
    _libc_free((unsigned __int64)v90);
    v14 = v77;
  }
LABEL_5:
  if ( v87 != v89 )
  {
    v81 = v14;
    _libc_free((unsigned __int64)v87);
    return v81;
  }
  return v14;
}
