// Function: sub_175CDB0
// Address: 0x175cdb0
//
__int64 __fastcall sub_175CDB0(
        _QWORD *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        double a7,
        double a8,
        double a9,
        __int64 *a10)
{
  __int64 *v10; // r15
  __int64 v14; // rcx
  __int64 *v15; // r8
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rdx
  int v19; // eax
  unsigned __int8 v20; // al
  __int64 v21; // rax
  unsigned int v22; // r8d
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // eax
  __int64 **v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int8 v33; // al
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rbx
  __int64 v37; // r9
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rbx
  bool v43; // al
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rbx
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rbx
  int v54; // eax
  bool v55; // al
  __int64 v56; // rax
  __int64 v57; // rdi
  __int64 v58; // rbx
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rbx
  __int64 v62; // rax
  __int64 v63; // rdi
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rax
  unsigned int v67; // edx
  __int64 v68; // rax
  char v69; // cl
  unsigned int v70; // edx
  bool v71; // al
  unsigned int i; // edx
  __int64 v73; // rax
  bool v74; // al
  unsigned int v75; // edx
  __int64 v76; // rax
  char v77; // cl
  unsigned int v78; // edx
  bool v79; // al
  int v80; // [rsp+Ch] [rbp-64h]
  int v81; // [rsp+Ch] [rbp-64h]
  int v82; // [rsp+Ch] [rbp-64h]
  __int64 *v83; // [rsp+10h] [rbp-60h]
  __int64 *v84; // [rsp+10h] [rbp-60h]
  int v85; // [rsp+10h] [rbp-60h]
  int v86; // [rsp+10h] [rbp-60h]
  unsigned int v87; // [rsp+10h] [rbp-60h]
  unsigned __int8 v88; // [rsp+10h] [rbp-60h]
  unsigned int v89; // [rsp+10h] [rbp-60h]
  unsigned int v90; // [rsp+10h] [rbp-60h]
  unsigned int v91; // [rsp+10h] [rbp-60h]
  __int64 v93[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v94; // [rsp+30h] [rbp-40h]

  v10 = (__int64 *)a3;
  v14 = (unsigned int)*(unsigned __int8 *)(a5 + 16) - 24;
  if ( (unsigned int)v14 <= 0x1C
    && ((1LL << (*(_BYTE *)(a5 + 16) - 24)) & 0x1C019800) != 0
    && *(_BYTE *)(a3 + 16) <= 0x10u
    && *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v10 = (__int64 *)a4;
    a4 = a3;
  }
  v15 = (__int64 *)a1[1];
  v15[1] = *(_QWORD *)(a5 + 40);
  v15[2] = a5 + 24;
  v16 = *(_QWORD *)(a5 + 48);
  v93[0] = v16;
  if ( v16 )
  {
    v83 = v15;
    sub_1623A60((__int64)v93, v16, 2);
    v15 = v83;
    v17 = *v83;
    if ( !*v83 )
      goto LABEL_6;
    goto LABEL_5;
  }
  v17 = *v15;
  if ( *v15 )
  {
LABEL_5:
    v84 = v15;
    sub_161E7C0((__int64)v15, v17);
    v15 = v84;
LABEL_6:
    v17 = v93[0];
    *v15 = v93[0];
    if ( v17 )
    {
      sub_1623210((__int64)v93, (unsigned __int8 *)v17, (__int64)v15);
    }
    else
    {
      v17 = v93[0];
      if ( v93[0] )
        sub_161E7C0((__int64)v93, v93[0]);
    }
  }
  v18 = a2;
  switch ( a2 )
  {
    case 0u:
      v17 = a4;
      v29 = sub_14C28B0(v10, (__int64 *)a4, a1[333], a1[330], a5, a1[332]);
      if ( v29 == 2 )
      {
        v59 = sub_159C540(*(__int64 **)(a1[1] + 24LL));
        v60 = a1[1];
        v61 = v59;
        v94 = 257;
        *a6 = (__int64)sub_17094A0(v60, (__int64)v10, a4, v93, 1u, 0, a7, a8, a9);
        *a10 = v61;
        sub_164B7C0(*a6, a5);
        return 1;
      }
      else
      {
        if ( v29 )
        {
          if ( *(_BYTE *)(a4 + 16) <= 0x10u )
            goto LABEL_27;
          return 0;
        }
        v51 = sub_159C4F0(*(__int64 **)(a1[1] + 24LL));
        v52 = a1[1];
        v94 = 257;
        v53 = v51;
        *a6 = (__int64)sub_17094A0(v52, (__int64)v10, a4, v93, 0, 0, a7, a8, a9);
        *a10 = v53;
        sub_164B7C0(*a6, a5);
        return 1;
      }
    case 1u:
      if ( *(_BYTE *)(a4 + 16) <= 0x10u )
      {
LABEL_27:
        if ( sub_1593BB0(a4, v17, v18, v14) )
          goto LABEL_24;
        if ( *(_BYTE *)(a4 + 16) == 13 )
        {
          if ( *(_DWORD *)(a4 + 32) <= 0x40u )
          {
            v43 = *(_QWORD *)(a4 + 24) == 0;
          }
          else
          {
            v86 = *(_DWORD *)(a4 + 32);
            v43 = v86 == (unsigned int)sub_16A57B0(a4 + 24);
          }
          if ( v43 )
            goto LABEL_24;
        }
        else if ( *(_BYTE *)(*(_QWORD *)a4 + 8LL) == 16 )
        {
          v50 = sub_15A1020((_BYTE *)a4, v17, v27, v28);
          if ( v50 && *(_BYTE *)(v50 + 16) == 13 )
          {
            if ( sub_13D01C0(v50 + 24) )
              goto LABEL_24;
          }
          else
          {
            v80 = *(_QWORD *)(*(_QWORD *)a4 + 32LL);
            if ( !v80 )
              goto LABEL_24;
            v67 = 0;
            while ( 1 )
            {
              v89 = v67;
              v68 = sub_15A0A60(a4, v67);
              if ( !v68 )
                break;
              v69 = *(_BYTE *)(v68 + 16);
              v70 = v89;
              if ( v69 != 9 )
              {
                if ( v69 != 13 )
                  break;
                v71 = sub_13D01C0(v68 + 24);
                v70 = v89;
                if ( !v71 )
                  break;
              }
              v67 = v70 + 1;
              if ( v80 == v67 )
                goto LABEL_24;
            }
          }
        }
        if ( a2 != 1 )
          return 0;
      }
      if ( (unsigned int)sub_14C38D0((__int64)v10, (__int64 *)a4, a1[333], a1[330], a5, a1[332]) != 2 )
        return 0;
      v44 = sub_159C540(*(__int64 **)(a1[1] + 24LL));
      v45 = a1[1];
      v46 = v44;
      v94 = 257;
      *a6 = (__int64)sub_17094A0(v45, (__int64)v10, a4, v93, 0, 1, a7, a8, a9);
      *a10 = v46;
      sub_164B7C0(*a6, a5);
      return 1;
    case 2u:
    case 3u:
      if ( *(_BYTE *)(a4 + 16) > 0x10u )
        goto LABEL_45;
      if ( sub_1593BB0(a4, v17, a2, v14) )
        goto LABEL_24;
      if ( *(_BYTE *)(a4 + 16) == 13 )
      {
        if ( *(_DWORD *)(a4 + 32) > 0x40u )
        {
          v85 = *(_DWORD *)(a4 + 32);
          if ( v85 != (unsigned int)sub_16A57B0(a4 + 24) )
            goto LABEL_45;
LABEL_24:
          v25 = sub_159C540(*(__int64 **)(a1[1] + 24LL));
          v22 = 1;
          *a6 = (__int64)v10;
          *a10 = v25;
          return v22;
        }
        if ( !*(_QWORD *)(a4 + 24) )
          goto LABEL_24;
      }
      else if ( *(_BYTE *)(*(_QWORD *)a4 + 8LL) == 16 )
      {
        v66 = sub_15A1020((_BYTE *)a4, v17, v23, v24);
        if ( !v66 || *(_BYTE *)(v66 + 16) != 13 )
        {
          v82 = *(_QWORD *)(*(_QWORD *)a4 + 32LL);
          if ( v82 )
          {
            v75 = 0;
            while ( 1 )
            {
              v91 = v75;
              v76 = sub_15A0A60(a4, v75);
              if ( !v76 )
                goto LABEL_45;
              v77 = *(_BYTE *)(v76 + 16);
              v78 = v91;
              if ( v77 != 9 )
              {
                if ( v77 != 13 )
                  goto LABEL_45;
                v79 = sub_13D01C0(v76 + 24);
                v78 = v91;
                if ( !v79 )
                  goto LABEL_45;
              }
              v75 = v78 + 1;
              if ( v82 == v75 )
                goto LABEL_24;
            }
          }
          goto LABEL_24;
        }
        if ( sub_13D01C0(v66 + 24) )
          goto LABEL_24;
      }
LABEL_45:
      v37 = a1[332];
      v38 = a1[330];
      v39 = a1[333];
      if ( a2 == 3 )
      {
        if ( (unsigned int)sub_14C2E40((__int64)v10, (__int64 *)a4, v39, v38, a5, v37) != 2 )
          return 0;
        v56 = sub_159C540(*(__int64 **)(a1[1] + 24LL));
        v57 = a1[1];
        v58 = v56;
        v94 = 257;
        *a6 = (__int64)sub_171D0D0(v57, (__int64)v10, a4, v93, 0, 1, a7, a8, a9);
        *a10 = v58;
        sub_164B7C0(*a6, a5);
        return 1;
      }
      else
      {
        if ( (unsigned int)sub_14C2B30(v10, (__int64 *)a4, v39, v38, a5, v37) != 2 )
          return 0;
        v40 = sub_159C540(*(__int64 **)(a1[1] + 24LL));
        v41 = a1[1];
        v42 = v40;
        v94 = 257;
        *a6 = (__int64)sub_171D0D0(v41, (__int64)v10, a4, v93, 1, 0, a7, a8, a9);
        *a10 = v42;
        sub_164B7C0(*a6, a5);
        return 1;
      }
    case 4u:
      v17 = a4;
      v19 = sub_14BBDC0(v10, a4, a1[333], a1[330], a5, a1[332]);
      if ( v19 == 2 )
      {
        v62 = sub_159C540(*(__int64 **)(a1[1] + 24LL));
        v63 = a1[1];
        v64 = v62;
        v94 = 257;
        *a6 = (__int64)sub_171D160(v63, (__int64)v10, a4, v93, 1, 0, a7, a8, a9);
        *a10 = v64;
        sub_164B7C0(*a6, a5);
        return 1;
      }
      else
      {
        if ( v19 )
          goto LABEL_16;
        v47 = sub_159C4F0(*(__int64 **)(a1[1] + 24LL));
        v48 = a1[1];
        v49 = v47;
        v94 = 257;
        *a6 = (__int64)sub_171D160(v48, (__int64)v10, a4, v93, 0, 0, a7, a8, a9);
        *a10 = v49;
        sub_164B7C0(*a6, a5);
        return 1;
      }
    case 5u:
LABEL_16:
      v20 = *(_BYTE *)(a4 + 16);
      if ( v20 == 9 )
      {
        v30 = (__int64 **)sub_1643320(*(_QWORD **)(a1[1] + 24LL));
        v31 = sub_1599EF0(v30);
        v22 = 1;
        *a6 = a4;
        *a10 = v31;
        return v22;
      }
      if ( v20 > 0x10u )
        goto LABEL_38;
      if ( sub_1593BB0(a4, v17, v18, v14) )
        goto LABEL_19;
      if ( *(_BYTE *)(a4 + 16) != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)a4 + 8LL) != 16 )
          goto LABEL_38;
        v32 = sub_15A1020((_BYTE *)a4, v17, v18, v14);
        if ( v32 && *(_BYTE *)(v32 + 16) == 13 )
        {
          if ( !sub_13D01C0(v32 + 24) )
            goto LABEL_38;
        }
        else
        {
          v81 = *(_QWORD *)(*(_QWORD *)a4 + 32LL);
          if ( v81 )
          {
            for ( i = 0; i != v81; i = v18 + 1 )
            {
              v17 = i;
              v90 = i;
              v73 = sub_15A0A60(a4, i);
              if ( !v73 )
                goto LABEL_38;
              v14 = *(unsigned __int8 *)(v73 + 16);
              v18 = v90;
              if ( (_BYTE)v14 != 9 )
              {
                if ( (_BYTE)v14 != 13 )
                  goto LABEL_38;
                v74 = sub_13D01C0(v73 + 24);
                v18 = v90;
                if ( !v74 )
                  goto LABEL_38;
              }
            }
          }
        }
LABEL_19:
        v21 = sub_159C540(*(__int64 **)(a1[1] + 24LL));
        v22 = 1;
        *a6 = a4;
        *a10 = v21;
        return v22;
      }
      v18 = *(unsigned int *)(a4 + 32);
      if ( (unsigned int)v18 <= 0x40 )
      {
        v55 = *(_QWORD *)(a4 + 24) == 0;
      }
      else
      {
        v87 = *(_DWORD *)(a4 + 32);
        v54 = sub_16A57B0(a4 + 24);
        v18 = v87;
        v55 = v87 == v54;
      }
      if ( v55 )
        goto LABEL_19;
LABEL_38:
      v33 = sub_17573B0((_BYTE *)a4, v17, v18, v14);
      if ( v33 )
      {
        v88 = v33;
        v65 = sub_159C540(*(__int64 **)(a1[1] + 24LL));
        v22 = v88;
        *a6 = (__int64)v10;
        *a10 = v65;
      }
      else if ( a2 == 5 && (unsigned int)sub_14C2C60(v10, (__int64 *)a4, a1[333], a1[330], a5, a1[332]) == 2 )
      {
        v34 = sub_159C540(*(__int64 **)(a1[1] + 24LL));
        v35 = a1[1];
        v36 = v34;
        v94 = 257;
        *a6 = (__int64)sub_171D160(v35, (__int64)v10, a4, v93, 0, 1, a7, a8, a9);
        *a10 = v36;
        sub_164B7C0(*a6, a5);
        return 1;
      }
      else
      {
        return 0;
      }
      return v22;
    default:
      return 0;
  }
}
