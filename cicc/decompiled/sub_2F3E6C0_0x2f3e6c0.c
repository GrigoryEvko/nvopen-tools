// Function: sub_2F3E6C0
// Address: 0x2f3e6c0
//
__int64 __fastcall sub_2F3E6C0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned int v4; // r12d
  __int64 v5; // rbx
  unsigned int v6; // eax
  __int64 v7; // r8
  __int64 v9; // r13
  unsigned int v10; // r8d
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rsi
  int v16; // eax
  int v17; // eax
  __int64 v18; // rax
  int v19; // r9d
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // r12
  int v25; // r12d
  int v26; // r13d
  __int64 v27; // rbx
  __int64 v28; // r13
  __int64 v29; // rcx
  __int64 v30; // r8
  int v31; // r14d
  __int64 v32; // r12
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned int v37; // eax
  int v38; // eax
  int v39; // eax
  int v40; // eax
  int v41; // eax
  int v42; // eax
  int v43; // eax
  int v44; // eax
  int v45; // eax
  int v46; // eax
  int v47; // eax
  int v48; // eax
  int v49; // eax
  int v50; // eax
  int v51; // eax
  int v52; // eax
  int v53; // eax
  int v54; // eax
  int v55; // eax
  int v56; // eax
  int v57; // eax
  int v58; // eax
  int v59; // eax
  int v60; // eax
  int v61; // eax
  int v62; // eax
  int v63; // eax
  __int64 (*v64)(); // rax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r8
  unsigned __int8 v68; // r9
  __int64 v69; // rcx
  __int64 v70; // rdi
  __int64 v71; // r10
  __int64 (*v72)(); // rax
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  unsigned __int16 v76; // ax
  unsigned int v77; // eax
  __int64 v78; // rbx
  int v79; // r8d
  int v80; // r12d
  __int64 v81; // rax
  __int64 v82; // [rsp+8h] [rbp-78h]
  unsigned int v83; // [rsp+14h] [rbp-6Ch]
  __int64 v84; // [rsp+18h] [rbp-68h]
  __int64 v85; // [rsp+20h] [rbp-60h]
  unsigned __int8 v86; // [rsp+20h] [rbp-60h]
  unsigned __int8 v87; // [rsp+28h] [rbp-58h]
  unsigned __int8 v88; // [rsp+28h] [rbp-58h]
  __int64 v89; // [rsp+28h] [rbp-58h]
  __int64 v90; // [rsp+30h] [rbp-50h]
  __int64 v91; // [rsp+30h] [rbp-50h]
  __int64 v92; // [rsp+30h] [rbp-50h]
  __int64 v93; // [rsp+30h] [rbp-50h]
  unsigned __int8 v94; // [rsp+38h] [rbp-48h]
  unsigned __int8 v95; // [rsp+38h] [rbp-48h]
  __int64 v96; // [rsp+38h] [rbp-48h]

  v3 = a2 + 24;
  v4 = 0;
  v5 = *(_QWORD *)(a2 + 32);
  if ( v5 != a2 + 24 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v5 )
          BUG();
        v6 = *(_DWORD *)(v5 - 20);
        v7 = v5 - 56;
        if ( v6 > 0x11A )
          break;
        if ( v6 <= 0x57 )
          goto LABEL_7;
        switch ( v6 )
        {
          case 0x58u:
          case 0x5Au:
            if ( !*(_QWORD *)(v5 - 40) )
              goto LABEL_7;
            v18 = v5;
            v19 = 0;
            v20 = *(_QWORD *)(v5 - 40);
            v82 = v7;
            v95 = v4;
            v21 = 0;
            v22 = v18;
            while ( 2 )
            {
              v24 = *(_QWORD *)(v20 + 24);
              if ( *(_BYTE *)v24 != 85 )
                goto LABEL_118;
              v92 = *(_QWORD *)(*(_QWORD *)(v24 - 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF)) + 8LL);
              if ( *(_BYTE *)(v92 + 8) != 18 )
                goto LABEL_118;
              v85 = v22;
              v88 = v19;
              v64 = *(__int64 (**)())(*(_QWORD *)*a1 + 16LL);
              if ( v64 == sub_23CE270 )
                BUG();
              v65 = ((__int64 (__fastcall *)(__int64, __int64))v64)(*a1, v82);
              v67 = 0;
              v68 = v88;
              v69 = v85;
              v70 = v65;
              v71 = v92;
              v72 = *(__int64 (**)())(*(_QWORD *)v65 + 144LL);
              if ( v72 != sub_2C8F680 )
              {
                v81 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))v72)(
                        v70,
                        v82,
                        v66,
                        v85,
                        0,
                        v88);
                v69 = v85;
                v68 = v88;
                v71 = v92;
                v67 = v81;
              }
              v86 = v68;
              v93 = v69;
              v89 = v71;
              v84 = v67;
              v83 = sub_2FEBF10(v67, *(unsigned int *)(v69 - 20));
              v76 = sub_30097B0(v89, 0, v73, v74, v75);
              v22 = v93;
              v19 = v86;
              if ( v76 )
              {
                a3 = v76;
                if ( *(_QWORD *)(v84 + 8LL * v76 + 112) )
                {
                  if ( v83 > 0x1F3 || *(_BYTE *)(v83 + 500LL * v76 + v84 + 6414) != 2 )
                    goto LABEL_118;
                }
              }
              v77 = sub_31A2800(a2, v24);
              v19 = v86;
              v22 = v93;
              a3 = v77;
              if ( !(_BYTE)v77 )
              {
LABEL_118:
                if ( !*(_QWORD *)(v22 - 40) )
                {
LABEL_28:
                  v25 = v95;
                  v26 = v19;
                  v27 = v22;
                  goto LABEL_29;
                }
LABEL_24:
                v23 = *(_QWORD *)(v20 + 8);
                if ( v23 )
                  goto LABEL_25;
                goto LABEL_28;
              }
              v23 = *(_QWORD *)(v93 - 40);
              if ( v23 )
              {
                v19 = a3;
                if ( !v21 )
                {
                  v20 = 0;
LABEL_25:
                  v21 = v20;
                  v20 = v23;
                  continue;
                }
                v20 = v21;
                goto LABEL_24;
              }
              break;
            }
            v25 = v95;
            v27 = v93;
            v26 = a3;
LABEL_29:
            v5 = *(_QWORD *)(v27 + 8);
            v4 = v26 | v25;
            if ( v3 == v5 )
              return v4;
            break;
          case 0x9Au:
          case 0xEEu:
          case 0xF0u:
          case 0xF1u:
          case 0xF3u:
          case 0xF5u:
            v17 = sub_2F3D720(a1, v5 - 56);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v17;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0xA4u:
          case 0xA5u:
          case 0xA6u:
          case 0xA7u:
          case 0xA8u:
            goto LABEL_10;
          case 0xCEu:
          case 0x11Au:
            v28 = *(_QWORD *)(v5 - 40);
            v29 = 0;
            v30 = 0;
            if ( !v28 )
              goto LABEL_7;
            v91 = v3;
            v31 = 0;
            v87 = v4;
            v32 = v5;
            v33 = 0;
            while ( 2 )
            {
              v35 = *(_QWORD *)(v28 + 24);
              if ( *(_BYTE *)v35 != 85
                || (v96 = *(_QWORD *)(*(_QWORD *)(v35 + 40) + 72LL),
                    v36 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))a1[3])(
                            a1[4],
                            v96,
                            a3,
                            v29,
                            v30),
                    v37 = sub_2891760(v96, v36, 0),
                    a3 = v37,
                    v31 |= v37,
                    !(_BYTE)v37) )
              {
                if ( *(_QWORD *)(v32 - 40) )
                  goto LABEL_34;
                goto LABEL_102;
              }
              v34 = *(_QWORD *)(v32 - 40);
              if ( v34 )
              {
                if ( !v33 )
                {
                  v28 = 0;
                  v31 = a3;
LABEL_35:
                  v33 = v28;
                  v28 = v34;
                  continue;
                }
                v28 = v33;
                v31 = a3;
LABEL_34:
                v34 = *(_QWORD *)(v28 + 8);
                if ( v34 )
                  goto LABEL_35;
LABEL_102:
                v78 = v32;
                v79 = v31;
                v80 = v87;
                v3 = v91;
                goto LABEL_103;
              }
              break;
            }
            v78 = v32;
            v3 = v91;
            v80 = v87;
            v79 = a3;
LABEL_103:
            v5 = *(_QWORD *)(v78 + 8);
            v4 = v79 | v80;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0xD6u:
            v38 = sub_2F3C710(v5 - 56);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v38;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0xFFu:
            v51 = sub_2F3CC20(v5 - 56, (__int64)"objc_autorelease", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v51;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x100u:
            v50 = sub_2F3CC20(v5 - 56, (__int64)"objc_autoreleasePoolPop", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v50;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x101u:
            v55 = sub_2F3CC20(v5 - 56, (__int64)"objc_autoreleasePoolPush", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v55;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x102u:
            v54 = sub_2F3CC20(v5 - 56, (__int64)"objc_autoreleaseReturnValue", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v54;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x105u:
            v53 = sub_2F3CC20(v5 - 56, (__int64)"objc_copyWeak", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v53;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x106u:
            v52 = sub_2F3CC20(v5 - 56, (__int64)"objc_destroyWeak", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v52;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x107u:
            v63 = sub_2F3CC20(v5 - 56, (__int64)"objc_initWeak", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v63;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x108u:
            v62 = sub_2F3CC20(v5 - 56, (__int64)"objc_loadWeak", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v62;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x109u:
            v61 = sub_2F3CC20(v5 - 56, (__int64)"objc_loadWeakRetained", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v61;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x10Au:
            v60 = sub_2F3CC20(v5 - 56, (__int64)"objc_moveWeak", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v60;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x10Bu:
            v59 = sub_2F3CC20(v5 - 56, (__int64)"objc_release", 1);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v59;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x10Cu:
            v58 = sub_2F3CC20(v5 - 56, (__int64)"objc_retain", 1);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v58;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x10Du:
            v57 = sub_2F3CC20(v5 - 56, (__int64)"objc_retain_autorelease", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v57;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x10Eu:
            v56 = sub_2F3CC20(v5 - 56, (__int64)"objc_retainAutorelease", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v56;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x10Fu:
            v49 = sub_2F3CC20(v5 - 56, (__int64)"objc_retainAutoreleaseReturnValue", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v49;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x110u:
            v48 = sub_2F3CC20(v5 - 56, (__int64)"objc_retainAutoreleasedReturnValue", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v48;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x111u:
            v47 = sub_2F3CC20(v5 - 56, (__int64)"objc_retainBlock", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v47;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x112u:
            v46 = sub_2F3CC20(v5 - 56, (__int64)"objc_retainedObject", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v46;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x113u:
            v45 = sub_2F3CC20(v5 - 56, (__int64)"objc_storeStrong", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v45;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x114u:
            v44 = sub_2F3CC20(v5 - 56, (__int64)"objc_storeWeak", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v44;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x115u:
            v43 = sub_2F3CC20(v5 - 56, (__int64)"objc_sync_enter", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v43;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x116u:
            v42 = sub_2F3CC20(v5 - 56, (__int64)"objc_sync_exit", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v42;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x117u:
            v41 = sub_2F3CC20(v5 - 56, (__int64)"objc_unretainedObject", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v41;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x118u:
            v40 = sub_2F3CC20(v5 - 56, (__int64)"objc_unretainedPointer", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v40;
            if ( v3 == v5 )
              return v4;
            continue;
          case 0x119u:
            v39 = sub_2F3CC20(v5 - 56, (__int64)"objc_unsafeClaimAutoreleasedReturnValue", 0);
            v5 = *(_QWORD *)(v5 + 8);
            v4 |= v39;
            if ( v3 == v5 )
              return v4;
            continue;
          default:
            goto LABEL_7;
        }
      }
      if ( v6 - 404 <= 0x58 )
      {
LABEL_10:
        v9 = *(_QWORD *)(v5 - 40);
        a3 = 0;
        if ( v9 )
        {
          v90 = v3;
          v10 = v4;
          v11 = v5;
          v12 = 0;
          while ( 2 )
          {
            v14 = *(_QWORD *)(v9 + 24);
            if ( *(_BYTE *)v14 == 85
              && (v94 = v10,
                  v15 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))a1[1])(
                          a1[2],
                          *(_QWORD *)(*(_QWORD *)(v14 + 40) + 72LL),
                          a3),
                  v16 = sub_34E4CC0(v14, v15),
                  LOBYTE(v15) = v16 != 0,
                  v10 = v15 | v94,
                  v16 == 2) )
            {
              v13 = *(_QWORD *)(v11 - 40);
              if ( !v13 )
              {
LABEL_6:
                v5 = v11;
                v3 = v90;
                v4 = v10;
                goto LABEL_7;
              }
              if ( !v12 )
              {
                v9 = 0;
LABEL_14:
                v12 = v9;
                v9 = v13;
                continue;
              }
              v9 = v12;
            }
            else if ( !*(_QWORD *)(v11 - 40) )
            {
              goto LABEL_6;
            }
            break;
          }
          v13 = *(_QWORD *)(v9 + 8);
          if ( !v13 )
            goto LABEL_6;
          goto LABEL_14;
        }
      }
LABEL_7:
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v3 != v5 );
  }
  return v4;
}
