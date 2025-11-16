// Function: sub_E246E0
// Address: 0xe246e0
//
unsigned __int64 __fastcall sub_E246E0(__int64 a1, size_t *a2)
{
  __int64 **v2; // r15
  _QWORD *v4; // r13
  __int64 v5; // r12
  size_t v6; // rax
  _BYTE *v7; // rdx
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  char v10; // r13
  _QWORD *v11; // rsi
  char v13; // al
  __int64 v14; // rdx
  __int64 v15; // r9
  char v16; // al
  __int64 *v17; // rax
  __int64 v18; // rax
  char v19; // dl
  _QWORD *v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdx
  size_t v24; // rax
  unsigned __int8 *v25; // rcx
  __int64 v26; // r13
  size_t v27; // rax
  __int64 v28; // r8
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // rax
  size_t v38; // rax
  unsigned __int64 *v39; // rax
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  _QWORD *v43; // rdx
  unsigned __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rcx
  char *v47; // rax
  char v48; // si
  __int64 v49; // r8
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // rax
  __int64 v53; // r13
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rax
  __int64 *v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // r8
  __int64 v63; // rax
  __int64 v64; // [rsp-8h] [rbp-98h]
  size_t v65; // [rsp+10h] [rbp-80h]
  _BYTE *v66; // [rsp+18h] [rbp-78h]
  unsigned __int64 v67; // [rsp+18h] [rbp-78h]
  char v68; // [rsp+20h] [rbp-70h]
  _QWORD *v69; // [rsp+20h] [rbp-70h]
  unsigned __int64 v70; // [rsp+20h] [rbp-70h]
  unsigned __int64 v71; // [rsp+20h] [rbp-70h]
  unsigned __int64 v72; // [rsp+20h] [rbp-70h]
  unsigned __int64 *v73; // [rsp+20h] [rbp-70h]
  char v74; // [rsp+20h] [rbp-70h]
  __int64 v75; // [rsp+20h] [rbp-70h]
  __int64 *v76; // [rsp+20h] [rbp-70h]
  __int64 v77; // [rsp+20h] [rbp-70h]
  _QWORD *v78; // [rsp+28h] [rbp-68h]
  _QWORD *v79; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v80[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v81[8]; // [rsp+50h] [rbp-40h] BYREF

  v2 = (__int64 **)(a1 + 16);
  v4 = &v79;
  v5 = 0;
  v79 = 0;
  while ( 1 )
  {
    do
    {
      v6 = *a2;
      if ( *a2 )
      {
        v7 = (_BYTE *)a2[1];
        if ( *v7 == 64 )
        {
          v11 = v79;
          a2[1] = (size_t)(v7 + 1);
          *a2 = v6 - 1;
          return sub_E208B0(v2, v11, v5);
        }
      }
    }
    while ( (unsigned __int8)sub_E20730(a2, 2u, "$S")
         || (unsigned __int8)sub_E20730(a2, 3u, "$$V")
         || (unsigned __int8)sub_E20730(a2, 4u, "$$$V")
         || (unsigned __int8)sub_E20730(a2, 3u, "$$Z") );
    v8 = *(_QWORD **)(a1 + 16);
    ++v5;
    v9 = (_QWORD *)((*v8 + v8[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v8[1] = (char *)v9 + 16LL - *v8;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v20 = (_QWORD *)sub_22077B0(32);
      if ( v20 )
      {
        *v20 = 0;
        v20[1] = 0;
        v20[2] = 0;
        v20[3] = 0;
      }
      v69 = v20;
      v78 = (_QWORD *)sub_2207820(4096);
      *v69 = v78;
      v21 = *(_QWORD *)(a1 + 16);
      v69[2] = 4096;
      v69[3] = v21;
      *(_QWORD *)(a1 + 16) = v69;
      v69[1] = 16;
      if ( v78 )
      {
        *v78 = 0;
        v78[1] = 0;
      }
    }
    else
    {
      v78 = 0;
      if ( v9 )
      {
        v78 = v9;
        *v9 = 0;
        v9[1] = 0;
      }
    }
    *v4 = v78;
    v10 = sub_E20730(a2, 2u, "$M");
    if ( v10 )
    {
      sub_E27700(a1, a2, 0);
      if ( *(_BYTE *)(a1 + 8) )
        return 0;
    }
    if ( (unsigned __int8)sub_E20730(a2, 3u, "$$Y") )
    {
      *v78 = sub_E27270(a1, a2);
      goto LABEL_15;
    }
    if ( (unsigned __int8)sub_E20730(a2, 3u, "$$B") )
      goto LABEL_32;
    v13 = sub_E20730(a2, 3u, "$$C");
    v14 = 1;
    if ( v13 )
      goto LABEL_33;
    v65 = *a2;
    v66 = (_BYTE *)a2[1];
    v68 = v10 ^ 1;
    if ( (unsigned __int8)sub_E206D0(*a2, v66, 2, (__int64)"$1", 1, (__int64)"1", v10 ^ 1u)
      || (unsigned __int8)sub_E206D0(v65, v66, 2, (__int64)"$H", 1, (__int64)"H", v68)
      || (unsigned __int8)sub_E206D0(v65, v66, 2, (__int64)"$I", 1, (__int64)"I", v68)
      || (v16 = sub_E206D0(v65, v66, 2, (__int64)"$J", 1, (__int64)"J", v68), v15 = v64, v16) )
    {
      v22 = *(_QWORD **)(a1 + 16);
      v23 = (*v22 + v22[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v22[1] = v23 + 64LL - *v22;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        v39 = (unsigned __int64 *)sub_22077B0(32);
        if ( v39 )
        {
          *v39 = 0;
          v39[1] = 0;
          v39[2] = 0;
          v39[3] = 0;
        }
        v73 = v39;
        v23 = sub_2207820(4096);
        *v73 = v23;
        v40 = *(_QWORD *)(a1 + 16);
        v73[2] = 4096;
        v73[3] = v40;
        *(_QWORD *)(a1 + 16) = v73;
        v73[1] = 64;
      }
      if ( !v23 )
      {
        *v78 = 0;
        MEMORY[0x3C] = 0;
        BUG();
      }
      *(_BYTE *)(v23 + 60) = 0;
      *(_DWORD *)(v23 + 8) = 21;
      *(_QWORD *)(v23 + 16) = 0;
      *(_QWORD *)v23 = &unk_49E0F38;
      *(_DWORD *)(v23 + 24) = 0;
      *(_DWORD *)(v23 + 56) = 0;
      *v78 = v23;
      *(_BYTE *)(v23 + 60) = 1;
      if ( v10 )
      {
        v24 = *a2;
        v25 = (unsigned __int8 *)a2[1];
      }
      else
      {
        v25 = (unsigned __int8 *)(a2[1] + 1);
        v38 = *a2;
        a2[1] = (size_t)v25;
        v24 = v38 - 1;
        *a2 = v24;
      }
      v26 = 0;
      v27 = v24 - 1;
      v28 = *v25;
      a2[1] = (size_t)(v25 + 1);
      *a2 = v27;
      if ( v27 && v25[1] == 63 )
      {
        v67 = v23;
        v74 = v28;
        v41 = sub_E25DD0(a1, a2, v23, v25, v28, v15);
        v26 = v41;
        if ( *(_BYTE *)(a1 + 8) || (v42 = *(_QWORD *)(v41 + 16)) == 0 )
        {
          *(_BYTE *)(a1 + 8) = 1;
          return 0;
        }
        sub_E21CC0(
          a1,
          *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v42 + 16) + 16LL) + 8LL * *(_QWORD *)(*(_QWORD *)(v42 + 16) + 24LL) - 8));
        LOBYTE(v28) = v74;
        v23 = v67;
      }
      if ( (_BYTE)v28 != 73 )
      {
        if ( (char)v28 <= 73 )
        {
          if ( (_BYTE)v28 != 49 )
            goto LABEL_47;
          goto LABEL_48;
        }
        v71 = v23;
        v32 = sub_E21AC0(a1, a2);
        v23 = v71;
        v33 = v32;
        v34 = *(int *)(v71 + 24);
        *(_DWORD *)(v71 + 24) = v34 + 1;
        *(_QWORD *)(v71 + 8 * v34 + 32) = v33;
      }
      v72 = v23;
      v35 = sub_E21AC0(a1, a2);
      v23 = v72;
      v36 = v35;
      v37 = *(int *)(v72 + 24);
      *(_DWORD *)(v72 + 24) = v37 + 1;
      *(_QWORD *)(v72 + 8 * v37 + 32) = v36;
LABEL_47:
      v70 = v23;
      v29 = sub_E21AC0(a1, a2);
      v23 = v70;
      v30 = v29;
      v31 = *(int *)(v70 + 24);
      *(_DWORD *)(v70 + 24) = v31 + 1;
      *(_QWORD *)(v70 + 8 * v31 + 32) = v30;
LABEL_48:
      *(_DWORD *)(v23 + 56) = 1;
      *(_QWORD *)(v23 + 16) = v26;
      goto LABEL_15;
    }
    if ( *a2 > 2 && *(_WORD *)v66 == 17700 && v66[2] == 63 )
    {
      sub_E20730(a2, 2u, "$E");
      v53 = sub_E245F0(v2);
      *v78 = v53;
      v58 = sub_E25DD0(a1, a2, v54, v55, v56, v57);
      *(_DWORD *)(v53 + 56) = 2;
      *(_QWORD *)(v53 + 16) = v58;
      goto LABEL_15;
    }
    if ( !(unsigned __int8)sub_E206D0(v65, v66, 2, (__int64)"$F", 1, (__int64)"F", v68)
      && !(unsigned __int8)sub_E206D0(v65, v66, 2, (__int64)"$G", 1, (__int64)"G", v68) )
    {
      v80[0] = 2;
      v80[1] = "$0";
      v81[1] = (__int64)"0";
      v17 = v80;
      if ( v10 == 1 )
        v17 = v81;
      v81[0] = 1;
      if ( (unsigned __int8)sub_E20730(a2, *v17, (const void *)v17[1]) )
      {
        LOBYTE(v80[0]) = 0;
        v81[0] = 0;
        v18 = sub_E219C0(a1, a2);
        LOBYTE(v80[0]) = v19;
        v81[0] = v18;
        *v78 = sub_E24500(v2, v81, (char *)v80);
        goto LABEL_15;
      }
LABEL_32:
      v14 = 0;
LABEL_33:
      *v78 = sub_E27700(a1, a2, v14);
      goto LABEL_15;
    }
    v43 = *(_QWORD **)(a1 + 16);
    v44 = (*v43 + v43[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v43[1] = v44 + 64LL - *v43;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v59 = (__int64 *)sub_22077B0(32);
      if ( v59 )
      {
        *v59 = 0;
        v59[1] = 0;
        v59[2] = 0;
        v59[3] = 0;
      }
      v76 = v59;
      v45 = sub_2207820(4096);
      *v76 = v45;
      v60 = *(_QWORD *)(a1 + 16);
      v76[2] = 4096;
      v76[3] = v60;
      *(_QWORD *)(a1 + 16) = v76;
      v76[1] = 64;
      if ( !v45 )
        goto LABEL_64;
      goto LABEL_63;
    }
    v45 = 0;
    if ( v44 )
    {
      v45 = v44;
LABEL_63:
      *(_DWORD *)(v45 + 8) = 21;
      *(_QWORD *)(v45 + 16) = 0;
      *(_DWORD *)(v45 + 24) = 0;
      *(_QWORD *)v45 = &unk_49E0F38;
      *(_DWORD *)(v45 + 56) = 0;
      *(_BYTE *)(v45 + 60) = 0;
    }
LABEL_64:
    *v78 = v45;
    if ( v10 )
    {
      v46 = *a2;
      v47 = (char *)a2[1];
    }
    else
    {
      v47 = (char *)(a2[1] + 1);
      v46 = *a2 - 1;
      a2[1] = (size_t)v47;
      *a2 = v46;
    }
    v48 = *v47;
    a2[1] = (size_t)(v47 + 1);
    *a2 = v46 - 1;
    if ( v48 != 70 )
    {
      v77 = v45;
      v61 = sub_E21AC0(a1, a2);
      v45 = v77;
      v62 = v61;
      v63 = *(int *)(v77 + 24);
      *(_DWORD *)(v77 + 24) = v63 + 1;
      *(_QWORD *)(v77 + 8 * v63 + 32) = v62;
    }
    v75 = v45;
    v49 = sub_E21AC0(a1, a2);
    v50 = *(int *)(v75 + 24);
    *(_DWORD *)(v75 + 24) = v50 + 1;
    *(_QWORD *)(v75 + 8 * v50 + 32) = v49;
    v51 = sub_E21AC0(a1, a2);
    v52 = *(int *)(v75 + 24);
    *(_DWORD *)(v75 + 24) = v52 + 1;
    *(_QWORD *)(v75 + 8 * v52 + 32) = v51;
    *(_BYTE *)(v75 + 60) = 1;
LABEL_15:
    if ( *(_BYTE *)(a1 + 8) )
      return 0;
    v4 = v78 + 1;
  }
}
