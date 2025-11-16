// Function: sub_17C63C0
// Address: 0x17c63c0
//
void __fastcall sub_17C63C0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // r14
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // r14
  _QWORD *v12; // rax
  __int64 v13; // r15
  _QWORD *v14; // rax
  __int64 *v15; // r10
  __int64 *v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 *v19; // rbx
  __int64 *v20; // r14
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // r9
  __int64 v25; // rbx
  __int64 v26; // rax
  _QWORD *v27; // r14
  _QWORD *v28; // rax
  _QWORD *v29; // rbx
  unsigned __int64 *v30; // r14
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rsi
  unsigned __int8 *v34; // rsi
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rdx
  unsigned __int8 *v41; // rsi
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rsi
  __int64 v46; // rdx
  unsigned __int8 *v47; // rsi
  __int64 *v48; // [rsp+8h] [rbp-138h]
  __int64 v49; // [rsp+10h] [rbp-130h]
  __int64 v50; // [rsp+10h] [rbp-130h]
  __int64 v51; // [rsp+10h] [rbp-130h]
  __int64 v52; // [rsp+18h] [rbp-128h]
  __int64 *v53; // [rsp+20h] [rbp-120h]
  __int64 v54; // [rsp+28h] [rbp-118h]
  __int64 v55; // [rsp+28h] [rbp-118h]
  __int64 v56; // [rsp+28h] [rbp-118h]
  __int64 v57; // [rsp+30h] [rbp-110h]
  __int64 v58; // [rsp+30h] [rbp-110h]
  __int64 v59; // [rsp+30h] [rbp-110h]
  __int64 v60; // [rsp+30h] [rbp-110h]
  __int64 v61; // [rsp+30h] [rbp-110h]
  __int64 *v62; // [rsp+30h] [rbp-110h]
  __int64 **v63; // [rsp+38h] [rbp-108h]
  __int64 v64; // [rsp+38h] [rbp-108h]
  __int64 v65; // [rsp+38h] [rbp-108h]
  __int64 v66; // [rsp+38h] [rbp-108h]
  __int64 v67; // [rsp+48h] [rbp-F8h] BYREF
  _QWORD v68[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v69[2]; // [rsp+60h] [rbp-E0h] BYREF
  __int16 v70; // [rsp+70h] [rbp-D0h]
  _QWORD v71[2]; // [rsp+80h] [rbp-C0h] BYREF
  __int16 v72; // [rsp+90h] [rbp-B0h]
  const char *v73; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v74; // [rsp+A8h] [rbp-98h]
  __int16 v75; // [rsp+B0h] [rbp-90h]
  __int64 *v76; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v77; // [rsp+C8h] [rbp-78h]
  __int64 *v78; // [rsp+D0h] [rbp-70h]
  _QWORD *v79; // [rsp+D8h] [rbp-68h]
  __int64 v80; // [rsp+E0h] [rbp-60h]
  int v81; // [rsp+E8h] [rbp-58h]
  __int64 v82; // [rsp+F0h] [rbp-50h]
  __int64 v83; // [rsp+F8h] [rbp-48h]

  v1 = a1;
  v53 = (__int64 *)sub_1643270(*(_QWORD **)a1[5]);
  v63 = (__int64 **)sub_16471D0(*(_QWORD **)a1[5], 0);
  v52 = sub_1643360(*(_QWORD **)a1[5]);
  v74 = 33;
  v2 = sub_16453E0(v53, 0);
  LOWORD(v78) = 261;
  v3 = a1[5];
  v73 = "__llvm_profile_register_functions";
  v76 = (__int64 *)&v73;
  v4 = sub_1648B60(120);
  v5 = v4;
  if ( v4 )
  {
    v57 = v4;
    sub_15E2490(v4, v2, 7, (__int64)&v76, v3);
    v5 = v57;
  }
  *(_BYTE *)(v5 + 32) = *(_BYTE *)(v5 + 32) & 0x3F | 0x80;
  if ( *(_BYTE *)a1 )
  {
    v61 = v5;
    sub_15E0D50(v5, -1, 28);
    v5 = v61;
  }
  v58 = v5;
  v76 = (__int64 *)v63;
  v6 = sub_1644EA0(v53, &v76, 1, 0);
  v7 = a1[5];
  v74 = 32;
  v8 = v6;
  v76 = (__int64 *)&v73;
  v73 = "__llvm_profile_register_function";
  v54 = v7;
  LOWORD(v78) = 261;
  v9 = sub_1648B60(120);
  v10 = v58;
  v11 = v9;
  if ( v9 )
  {
    sub_15E2490(v9, v8, 0, (__int64)&v76, v54);
    v10 = v58;
  }
  v55 = v10;
  v75 = 257;
  v59 = *(_QWORD *)a1[5];
  v12 = (_QWORD *)sub_22077B0(64);
  v13 = (__int64)v12;
  if ( v12 )
    sub_157FB60(v12, v59, (__int64)&v73, v55, 0);
  v14 = (_QWORD *)sub_157E9C0(v13);
  v15 = (__int64 *)a1[18];
  v16 = (__int64 *)a1[19];
  v77 = v13;
  v79 = v14;
  v17 = a1[24];
  v76 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v78 = (__int64 *)(v13 + 40);
  if ( v15 != v16 )
  {
    v18 = v11;
    v19 = v15;
    v20 = v16;
    do
    {
      while ( 1 )
      {
        v21 = *v19;
        if ( *v19 != v17 )
        {
          if ( *(_BYTE *)(v21 + 16) )
            break;
        }
        if ( v20 == ++v19 )
          goto LABEL_18;
      }
      v72 = 257;
      v70 = 257;
      if ( v63 != *(__int64 ***)v21 )
      {
        if ( *(_BYTE *)(v21 + 16) > 0x10u )
        {
          v75 = 257;
          v35 = sub_15FDBD0(47, v21, (__int64)v63, (__int64)&v73, 0);
          v36 = v35;
          if ( v77 )
          {
            v49 = v35;
            v48 = v78;
            sub_157E9D0(v77 + 40, v35);
            v36 = v49;
            v37 = *(_QWORD *)(v49 + 24);
            v38 = *v48;
            *(_QWORD *)(v49 + 32) = v48;
            v38 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v49 + 24) = v38 | v37 & 7;
            *(_QWORD *)(v38 + 8) = v49 + 24;
            *v48 = *v48 & 7 | (v49 + 24);
          }
          v50 = v36;
          sub_164B780(v36, v69);
          v21 = v50;
          if ( v76 )
          {
            v68[0] = v76;
            sub_1623A60((__int64)v68, (__int64)v76, 2);
            v21 = v50;
            v39 = *(_QWORD *)(v50 + 48);
            v40 = v50 + 48;
            if ( v39 )
            {
              sub_161E7C0(v50 + 48, v39);
              v21 = v50;
              v40 = v50 + 48;
            }
            v41 = (unsigned __int8 *)v68[0];
            *(_QWORD *)(v21 + 48) = v68[0];
            if ( v41 )
            {
              v51 = v21;
              sub_1623210((__int64)v68, v41, v40);
              v21 = v51;
            }
          }
        }
        else
        {
          v21 = sub_15A46C0(47, (__int64 ***)v21, v63, 0);
        }
      }
      v73 = (const char *)v21;
      ++v19;
      sub_1285290((__int64 *)&v76, *(_QWORD *)(v18 + 24), v18, (int)&v73, 1, (__int64)v71, 0);
      v17 = a1[24];
    }
    while ( v20 != v19 );
LABEL_18:
    v1 = a1;
  }
  if ( v17 )
  {
    v68[0] = v63;
    v68[1] = v52;
    v56 = sub_1644EA0(v53, v68, 2, 0);
    v60 = v1[5];
    v71[0] = "__llvm_profile_register_names_function";
    v71[1] = 38;
    v75 = 261;
    v73 = (const char *)v71;
    v22 = sub_1648B60(120);
    v23 = v22;
    if ( v22 )
      sub_15E2490(v22, v56, 0, (__int64)&v73, v60);
    v24 = v1[24];
    v72 = 257;
    v70 = 257;
    if ( v63 != *(__int64 ***)v24 )
    {
      if ( *(_BYTE *)(v24 + 16) > 0x10u )
      {
        v75 = 257;
        v42 = sub_15FDBD0(47, v24, (__int64)v63, (__int64)&v73, 0);
        if ( v77 )
        {
          v64 = v42;
          v62 = v78;
          sub_157E9D0(v77 + 40, v42);
          v42 = v64;
          v43 = *(_QWORD *)(v64 + 24);
          v44 = *v62;
          *(_QWORD *)(v64 + 32) = v62;
          v44 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v64 + 24) = v44 | v43 & 7;
          *(_QWORD *)(v44 + 8) = v64 + 24;
          *v62 = *v62 & 7 | (v64 + 24);
        }
        v65 = v42;
        sub_164B780(v42, v69);
        v24 = v65;
        if ( v76 )
        {
          v67 = (__int64)v76;
          sub_1623A60((__int64)&v67, (__int64)v76, 2);
          v24 = v65;
          v45 = *(_QWORD *)(v65 + 48);
          v46 = v65 + 48;
          if ( v45 )
          {
            sub_161E7C0(v65 + 48, v45);
            v24 = v65;
            v46 = v65 + 48;
          }
          v47 = (unsigned __int8 *)v67;
          *(_QWORD *)(v24 + 48) = v67;
          if ( v47 )
          {
            v66 = v24;
            sub_1623210((__int64)&v67, v47, v46);
            v24 = v66;
          }
        }
      }
      else
      {
        v24 = sub_15A46C0(47, (__int64 ***)v24, v63, 0);
      }
    }
    v25 = v1[25];
    v73 = (const char *)v24;
    v26 = sub_1643360(v79);
    v74 = sub_159C470(v26, v25, 0);
    sub_1285290((__int64 *)&v76, *(_QWORD *)(v23 + 24), v23, (int)&v73, 2, (__int64)v71, 0);
  }
  v27 = v79;
  v75 = 257;
  v28 = sub_1648A60(56, 0);
  v29 = v28;
  if ( v28 )
    sub_15F6F90((__int64)v28, (__int64)v27, 0, 0);
  if ( v77 )
  {
    v30 = (unsigned __int64 *)v78;
    sub_157E9D0(v77 + 40, (__int64)v29);
    v31 = v29[3];
    v32 = *v30;
    v29[4] = v30;
    v32 &= 0xFFFFFFFFFFFFFFF8LL;
    v29[3] = v32 | v31 & 7;
    *(_QWORD *)(v32 + 8) = v29 + 3;
    *v30 = *v30 & 7 | (unsigned __int64)(v29 + 3);
  }
  sub_164B780((__int64)v29, (__int64 *)&v73);
  if ( v76 )
  {
    v71[0] = v76;
    sub_1623A60((__int64)v71, (__int64)v76, 2);
    v33 = v29[6];
    if ( v33 )
      sub_161E7C0((__int64)(v29 + 6), v33);
    v34 = (unsigned __int8 *)v71[0];
    v29[6] = v71[0];
    if ( v34 )
      sub_1623210((__int64)v71, v34, (__int64)(v29 + 6));
    if ( v76 )
      sub_161E7C0((__int64)&v76, (__int64)v76);
  }
}
