// Function: sub_17D0680
// Address: 0x17d0680
//
void __fastcall sub_17D0680(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rax
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 *v6; // r13
  __int64 v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // r15
  __int64 v11; // r10
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 **v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rax
  _QWORD *v24; // r13
  _QWORD *v25; // r13
  __int64 *v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // r13
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // r13
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 *v34; // rax
  __int64 **v35; // rdx
  __int64 v36; // r10
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 *v39; // r8
  __int64 v40; // rsi
  __int64 v41; // rdx
  unsigned __int8 *v42; // rsi
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 *v46; // r8
  __int64 v47; // rdx
  unsigned __int8 *v48; // rsi
  unsigned __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 *v52; // r8
  __int64 v53; // rdx
  unsigned __int8 *v54; // rsi
  __int64 v55; // r9
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 *v58; // [rsp-150h] [rbp-150h]
  __int64 *v59; // [rsp-148h] [rbp-148h]
  _QWORD *v60; // [rsp-148h] [rbp-148h]
  __int64 *v61; // [rsp-148h] [rbp-148h]
  _QWORD *v62; // [rsp-140h] [rbp-140h]
  __int64 v63; // [rsp-140h] [rbp-140h]
  __int64 v64; // [rsp-138h] [rbp-138h]
  _QWORD *v65; // [rsp-130h] [rbp-130h]
  _QWORD *v66; // [rsp-130h] [rbp-130h]
  __int64 v67; // [rsp-130h] [rbp-130h]
  __int64 v68; // [rsp-130h] [rbp-130h]
  __int64 *v69; // [rsp-130h] [rbp-130h]
  __int64 v70; // [rsp-120h] [rbp-120h]
  _QWORD *v71; // [rsp-120h] [rbp-120h]
  __int64 **v72; // [rsp-120h] [rbp-120h]
  __int64 v73; // [rsp-120h] [rbp-120h]
  _BYTE *v74; // [rsp-120h] [rbp-120h]
  __int64 **v75; // [rsp-120h] [rbp-120h]
  __int64 v76; // [rsp-120h] [rbp-120h]
  __int64 v77; // [rsp-120h] [rbp-120h]
  __int64 v78; // [rsp-110h] [rbp-110h] BYREF
  __int64 v79[2]; // [rsp-108h] [rbp-108h] BYREF
  __int16 v80; // [rsp-F8h] [rbp-F8h]
  _BYTE v81[16]; // [rsp-E8h] [rbp-E8h] BYREF
  __int16 v82; // [rsp-D8h] [rbp-D8h]
  __int64 v83[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int16 v84; // [rsp-B8h] [rbp-B8h]
  __int64 v85[2]; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v86; // [rsp-98h] [rbp-98h]
  __int64 v87; // [rsp-88h] [rbp-88h] BYREF
  __int64 v88; // [rsp-80h] [rbp-80h]
  __int64 *v89; // [rsp-78h] [rbp-78h]
  _QWORD *v90; // [rsp-70h] [rbp-70h]

  if ( *(_DWORD *)(a1 + 56) )
  {
    v1 = sub_157ED20(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 480LL));
    sub_17CE510((__int64)&v87, v1, 0, 0, 0);
    v86 = 257;
    v2 = sub_156E5B0(&v87, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 232LL), (__int64)v85);
    *(_QWORD *)(a1 + 40) = v2;
    v3 = (__int64)v2;
    v4 = *(_QWORD *)(a1 + 16);
    v86 = 257;
    v5 = sub_15A0680(*(_QWORD *)(v4 + 176), 176, 0);
    v6 = (__int64 *)sub_12899C0(&v87, v5, v3, (__int64)v85, 0, 0);
    v7 = *(_QWORD *)(a1 + 16);
    v86 = 257;
    v8 = (_QWORD *)sub_1643330(*(_QWORD **)(v7 + 168));
    v9 = sub_17CEAE0(&v87, v8, (__int64)v6, v85);
    *(_QWORD *)(a1 + 32) = v9;
    sub_15E7430(&v87, v9, 8u, *(_QWORD **)(*(_QWORD *)(a1 + 16) + 224LL), 8u, v6, 0, 0, 0, 0, 0);
    sub_17CD270(&v87);
    v64 = *(unsigned int *)(a1 + 56);
    if ( *(_DWORD *)(a1 + 56) )
    {
      v10 = 0;
      do
      {
        v31 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v10);
        v32 = *(_QWORD *)(v31 + 32);
        if ( v32 == *(_QWORD *)(v31 + 40) + 40LL || !v32 )
          v33 = 0;
        else
          v33 = v32 - 24;
        sub_17CE510((__int64)&v87, v33, 0, 0, 0);
        v22 = *(_QWORD *)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
        v84 = 257;
        v34 = (__int64 *)sub_1647230(*(_QWORD **)(*(_QWORD *)(a1 + 16) + 168LL), 0);
        v75 = (__int64 **)sub_1646BA0(v34, 0);
        v82 = 257;
        v68 = sub_15A0680(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL), 16, 0);
        v80 = 257;
        v35 = *(__int64 ***)(*(_QWORD *)(a1 + 16) + 176LL);
        if ( v35 == *(__int64 ***)v22 )
        {
          v11 = v22;
        }
        else if ( *(_BYTE *)(v22 + 16) <= 0x10u )
        {
          v11 = sub_15A46C0(45, (__int64 ***)v22, v35, 0);
        }
        else
        {
          v86 = 257;
          v36 = sub_15FDBD0(45, v22, (__int64)v35, (__int64)v85, 0);
          if ( v88 )
          {
            v63 = v36;
            v59 = v89;
            sub_157E9D0(v88 + 40, v36);
            v36 = v63;
            v37 = *(_QWORD *)(v63 + 24);
            v38 = *v59;
            *(_QWORD *)(v63 + 32) = v59;
            v38 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v63 + 24) = v38 | v37 & 7;
            *(_QWORD *)(v38 + 8) = v63 + 24;
            *v59 = *v59 & 7 | (v63 + 24);
          }
          v62 = (_QWORD *)v36;
          sub_164B780(v36, v79);
          v11 = (__int64)v62;
          if ( v87 )
          {
            v60 = v62;
            v78 = v87;
            sub_1623A60((__int64)&v78, v87, 2);
            v11 = (__int64)v62;
            v39 = &v78;
            v40 = v62[6];
            v41 = (__int64)(v62 + 6);
            if ( v40 )
            {
              v58 = v62;
              v62 += 6;
              sub_161E7C0((__int64)(v60 + 6), v40);
              v39 = v58;
              v11 = (__int64)v60;
              v41 = (__int64)(v60 + 6);
            }
            v42 = (unsigned __int8 *)v78;
            *(_QWORD *)(v11 + 48) = v78;
            if ( v42 )
            {
              v62 = (_QWORD *)v11;
              sub_1623210((__int64)v39, v42, v41);
              v11 = (__int64)v62;
            }
          }
        }
        v12 = sub_12899C0(&v87, v11, v68, (__int64)v81, 0, 0);
        v13 = v12;
        if ( v75 != *(__int64 ***)v12 )
        {
          if ( *(_BYTE *)(v12 + 16) > 0x10u )
          {
            v86 = 257;
            v55 = sub_15FDBD0(46, v12, (__int64)v75, (__int64)v85, 0);
            if ( v88 )
            {
              v76 = v55;
              v69 = v89;
              sub_157E9D0(v88 + 40, v55);
              v55 = v76;
              v56 = *(_QWORD *)(v76 + 24);
              v57 = *v69;
              *(_QWORD *)(v76 + 32) = v69;
              v57 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v76 + 24) = v57 | v56 & 7;
              *(_QWORD *)(v57 + 8) = v76 + 24;
              *v69 = *v69 & 7 | (v76 + 24);
            }
            v77 = v55;
            sub_164B780(v55, v83);
            sub_12A86E0(&v87, v77);
            v13 = v77;
          }
          else
          {
            v13 = sub_15A46C0(46, (__int64 ***)v12, v75, 0);
          }
        }
        v86 = 257;
        v65 = sub_156E5B0(&v87, v13, (__int64)v85);
        v70 = *(_QWORD *)(a1 + 24);
        v14 = (__int64 *)sub_1643330(v90);
        v66 = (_QWORD *)sub_17CFB40(v70, (__int64)v65, &v87, v14, 0x10u);
        v71 = *(_QWORD **)(a1 + 32);
        v15 = sub_1643360(v90);
        v16 = (__int64 *)sub_159C470(v15, 176, 0);
        sub_15E7430(&v87, v66, 0x10u, v71, 0x10u, v16, 0, 0, 0, 0, 0);
        v17 = *(_QWORD *)(a1 + 16);
        v84 = 257;
        v18 = (__int64 *)sub_1647230(*(_QWORD **)(v17 + 168), 0);
        v72 = (__int64 **)sub_1646BA0(v18, 0);
        v19 = *(_QWORD *)(a1 + 16);
        v82 = 257;
        v67 = sub_15A0680(*(_QWORD *)(v19 + 176), 8, 0);
        v20 = *(_QWORD *)(a1 + 16);
        v80 = 257;
        v21 = *(__int64 ***)(v20 + 176);
        if ( v21 != *(__int64 ***)v22 )
        {
          if ( *(_BYTE *)(v22 + 16) > 0x10u )
          {
            v86 = 257;
            v22 = sub_15FDBD0(45, v22, (__int64)v21, (__int64)v85, 0);
            if ( v88 )
            {
              v62 = v89;
              sub_157E9D0(v88 + 40, v22);
              v43 = *v62;
              v44 = *(_QWORD *)(v22 + 24) & 7LL;
              *(_QWORD *)(v22 + 32) = v62;
              v43 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v22 + 24) = v43 | v44;
              *(_QWORD *)(v43 + 8) = v22 + 24;
              *v62 = *v62 & 7LL | (v22 + 24);
            }
            sub_164B780(v22, v79);
            if ( v87 )
            {
              v78 = v87;
              sub_1623A60((__int64)&v78, v87, 2);
              v45 = *(_QWORD *)(v22 + 48);
              v46 = &v78;
              v47 = v22 + 48;
              if ( v45 )
              {
                v61 = v62;
                v62 = (_QWORD *)(v22 + 48);
                sub_161E7C0(v22 + 48, v45);
                v46 = v61;
                v47 = v22 + 48;
              }
              v48 = (unsigned __int8 *)v78;
              *(_QWORD *)(v22 + 48) = v78;
              if ( v48 )
                sub_1623210((__int64)v46, v48, v47);
            }
          }
          else
          {
            v22 = sub_15A46C0(45, (__int64 ***)v22, v21, 0);
          }
        }
        v23 = sub_12899C0(&v87, v22, v67, (__int64)v81, 0, 0);
        v24 = (_QWORD *)v23;
        if ( v72 != *(__int64 ***)v23 )
        {
          if ( *(_BYTE *)(v23 + 16) > 0x10u )
          {
            v86 = 257;
            v24 = (_QWORD *)sub_15FDBD0(46, v23, (__int64)v72, (__int64)v85, 0);
            if ( v88 )
            {
              v72 = (__int64 **)v89;
              sub_157E9D0(v88 + 40, (__int64)v24);
              v49 = (unsigned __int64)*v72;
              v50 = v24[3] & 7LL;
              v24[4] = v72;
              v49 &= 0xFFFFFFFFFFFFFFF8LL;
              v24[3] = v49 | v50;
              *(_QWORD *)(v49 + 8) = v24 + 3;
              *v72 = (__int64 *)((unsigned __int64)*v72 & 7 | (unsigned __int64)(v24 + 3));
            }
            sub_164B780((__int64)v24, v83);
            if ( v87 )
            {
              v78 = v87;
              sub_1623A60((__int64)&v78, v87, 2);
              v51 = v24[6];
              v52 = &v78;
              v53 = (__int64)(v24 + 6);
              if ( v51 )
              {
                sub_161E7C0((__int64)(v24 + 6), v51);
                v52 = (__int64 *)v72;
                v53 = (__int64)(v24 + 6);
              }
              v54 = (unsigned __int8 *)v78;
              v24[6] = v78;
              if ( v54 )
                sub_1623210((__int64)v52, v54, v53);
            }
          }
          else
          {
            v24 = (_QWORD *)sub_15A46C0(46, (__int64 ***)v23, v72, 0);
          }
        }
        v86 = 257;
        v25 = sub_156E5B0(&v87, (__int64)v24, (__int64)v85);
        v73 = *(_QWORD *)(a1 + 24);
        v26 = (__int64 *)sub_1643330(v90);
        v27 = sub_17CFB40(v73, (__int64)v25, &v87, v26, 0x10u);
        v86 = 257;
        v28 = (_QWORD *)v27;
        v74 = *(_BYTE **)(a1 + 32);
        v29 = sub_1643330(v90);
        v30 = (_QWORD *)sub_17CE2F0((__int64)&v87, v29, v74, 0xB0u, v85);
        sub_15E7430(&v87, v28, 0x10u, v30, 0x10u, *(__int64 **)(a1 + 40), 0, 0, 0, 0, 0);
        if ( v87 )
          sub_161E7C0((__int64)&v87, v87);
        ++v10;
      }
      while ( v10 != v64 );
    }
  }
}
