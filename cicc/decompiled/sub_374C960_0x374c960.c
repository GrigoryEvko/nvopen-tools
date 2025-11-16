// Function: sub_374C960
// Address: 0x374c960
//
__int64 __fastcall sub_374C960(__int64 a1, __int64 *a2, unsigned __int8 a3)
{
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 *v7; // rdi
  __int64 v8; // r9
  unsigned int v9; // r12d
  __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // r11
  unsigned int v13; // r14d
  __int64 (__fastcall *v14)(__int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // ebx
  unsigned int v18; // r15d
  int v19; // r14d
  int v20; // eax
  __int64 (__fastcall *v22)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v23; // rax
  unsigned __int64 v24; // r14
  __int64 v25; // r11
  __int64 v26; // rax
  __int64 (__fastcall *v27)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v28; // rax
  unsigned __int64 v29; // r14
  __int64 v30; // r11
  __int64 (__fastcall *v31)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // r11
  __int64 (__fastcall *v35)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r11
  __int64 (__fastcall *v39)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rcx
  __int64 v42; // r11
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int128 v53; // [rsp-10h] [rbp-190h]
  __int64 v54; // [rsp-10h] [rbp-190h]
  __int64 v55; // [rsp-8h] [rbp-188h]
  unsigned __int64 v56; // [rsp+8h] [rbp-178h]
  unsigned __int64 v57; // [rsp+8h] [rbp-178h]
  __int64 v58; // [rsp+10h] [rbp-170h]
  __int64 v59; // [rsp+18h] [rbp-168h]
  __int64 v60; // [rsp+20h] [rbp-160h]
  __int64 v61; // [rsp+30h] [rbp-150h]
  __int64 v62; // [rsp+38h] [rbp-148h]
  __int64 v63; // [rsp+38h] [rbp-148h]
  __int64 v64; // [rsp+38h] [rbp-148h]
  __int64 v65; // [rsp+38h] [rbp-148h]
  __int64 v66; // [rsp+40h] [rbp-140h]
  __int64 v67; // [rsp+50h] [rbp-130h]
  __int64 *v69; // [rsp+60h] [rbp-120h]
  __int64 *v70; // [rsp+70h] [rbp-110h]
  __int64 v71; // [rsp+78h] [rbp-108h]
  int v72; // [rsp+78h] [rbp-108h]
  unsigned __int16 v73; // [rsp+8Ah] [rbp-F6h] BYREF
  unsigned int v74; // [rsp+8Ch] [rbp-F4h] BYREF
  __int64 v75; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v76; // [rsp+98h] [rbp-E8h]
  __int64 v77; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned __int64 v78; // [rsp+A8h] [rbp-D8h]
  __int64 v79; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned __int64 v80; // [rsp+B8h] [rbp-C8h]
  __int64 v81; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v82; // [rsp+C8h] [rbp-B8h]
  __int64 v83; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v84; // [rsp+D8h] [rbp-A8h]
  __int64 v85; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v86; // [rsp+E8h] [rbp-98h]
  unsigned __int64 v87; // [rsp+F0h] [rbp-90h]
  __int64 *v88; // [rsp+100h] [rbp-80h] BYREF
  __int64 v89; // [rsp+108h] [rbp-78h]
  _BYTE v90[112]; // [rsp+110h] [rbp-70h] BYREF

  v4 = *(__int64 **)(a1 + 8);
  v88 = (__int64 *)v90;
  v89 = 0x400000000LL;
  v5 = sub_2E79000(v4);
  v6 = *(_QWORD *)(a1 + 16);
  LOBYTE(v86) = 0;
  *((_QWORD *)&v53 + 1) = v86;
  v85 = 0;
  *(_QWORD *)&v53 = 0;
  sub_34B8C80(v6, v5, (__int64)a2, (__int64)&v88, 0, 0, v53);
  v7 = v88;
  v8 = v55;
  v69 = &v88[2 * (unsigned int)v89];
  if ( v69 != v88 )
  {
    v70 = v88;
    v9 = 0;
    while ( 1 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *v70;
      v71 = v70[1];
      v12 = *a2;
      v75 = v11;
      v76 = v71;
      if ( !(_WORD)v11 )
        break;
      v13 = *(unsigned __int16 *)(v10 + 2LL * (unsigned __int16)v11 + 2852);
LABEL_5:
      v14 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v10 + 736LL);
      BYTE2(v85) = 0;
      v72 = v14(v10, v12, v11, v71, v85, v8);
      if ( v72 )
      {
        v17 = 0;
        v18 = v13;
        v19 = v9;
        do
        {
          v20 = sub_374C900(a1, v18, a3, v15, v16, v8);
          if ( !v19 )
            v19 = v20;
          ++v17;
        }
        while ( v72 != v17 );
        v9 = v19;
      }
      v70 += 2;
      if ( v69 == v70 )
      {
        v7 = v88;
        goto LABEL_13;
      }
    }
    v67 = v12;
    if ( sub_30070B0((__int64)&v75) )
    {
      LOWORD(v85) = 0;
      LOWORD(v81) = 0;
      v86 = 0;
      sub_2FE8D10(v10, v67, (unsigned int)v75, v76, &v85, (unsigned int *)&v83, (unsigned __int16 *)&v81);
      v13 = (unsigned __int16)v81;
      v10 = *(_QWORD *)(a1 + 16);
      v12 = *a2;
      goto LABEL_5;
    }
    if ( !sub_3007070((__int64)&v75) )
      goto LABEL_55;
    v22 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v10 + 592LL);
    if ( v22 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v85, v10, v67, v11, v71);
      v23 = v66;
      LOWORD(v23) = v86;
      v24 = v87;
      v25 = v67;
      v66 = v23;
    }
    else
    {
      v43 = v22(v10, v67, v75, v76);
      v25 = v67;
      v66 = v43;
      v24 = v44;
    }
    v78 = v24;
    v26 = (unsigned __int16)v66;
    v77 = v66;
    if ( !(_WORD)v66 )
    {
      v62 = v25;
      if ( sub_30070B0((__int64)&v77) )
      {
        v86 = 0;
        LOWORD(v85) = 0;
        LOWORD(v81) = 0;
        sub_2FE8D10(v10, v62, (unsigned int)v77, v24, &v85, (unsigned int *)&v83, (unsigned __int16 *)&v81);
        goto LABEL_46;
      }
      if ( !sub_3007070((__int64)&v77) )
        goto LABEL_55;
      v27 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v10 + 592LL);
      if ( v27 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v85, v10, v62, v77, v78);
        v28 = v61;
        LOWORD(v28) = v86;
        v29 = v87;
        v30 = v62;
        v61 = v28;
      }
      else
      {
        v45 = v27(v10, v62, v77, v24);
        v30 = v62;
        v61 = v45;
        v29 = v46;
      }
      v80 = v29;
      v26 = (unsigned __int16)v61;
      v79 = v61;
      if ( !(_WORD)v61 )
      {
        v63 = v30;
        if ( !sub_30070B0((__int64)&v79) )
        {
          if ( !sub_3007070((__int64)&v79) )
            goto LABEL_55;
          v31 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v10 + 592LL);
          if ( v31 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v85, v10, v63, v79, v80);
            v32 = v60;
            LOWORD(v32) = v86;
            v33 = v87;
            v34 = v63;
            v60 = v32;
          }
          else
          {
            v47 = v31(v10, v63, v79, v29);
            v34 = v63;
            v60 = v47;
            v33 = v48;
          }
          v82 = v33;
          v26 = (unsigned __int16)v60;
          v81 = v60;
          if ( !(_WORD)v60 )
          {
            v56 = v33;
            v64 = v34;
            if ( sub_30070B0((__int64)&v81) )
            {
              LOWORD(v85) = 0;
              LOWORD(v74) = 0;
              v86 = 0;
              sub_2FE8D10(v10, v64, (unsigned int)v81, v56, &v85, (unsigned int *)&v83, (unsigned __int16 *)&v74);
              v8 = v54;
              v13 = (unsigned __int16)v74;
              goto LABEL_42;
            }
            if ( !sub_3007070((__int64)&v81) )
              goto LABEL_55;
            v35 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v10 + 592LL);
            if ( v35 == sub_2D56A50 )
            {
              sub_2FE6CC0((__int64)&v85, v10, v64, v81, v82);
              v36 = v58;
              LOWORD(v36) = v86;
              v37 = v87;
              v38 = v64;
              v58 = v36;
            }
            else
            {
              v49 = v35(v10, v64, v81, v56);
              v38 = v64;
              v58 = v49;
              v37 = v50;
            }
            v84 = v37;
            v26 = (unsigned __int16)v58;
            v83 = v58;
            if ( !(_WORD)v58 )
            {
              v57 = v37;
              v65 = v38;
              if ( sub_30070B0((__int64)&v83) )
              {
                LOWORD(v85) = 0;
                v73 = 0;
                v86 = 0;
                sub_2FE8D10(v10, v65, (unsigned int)v83, v57, &v85, &v74, &v73);
                v13 = v73;
              }
              else
              {
                if ( !sub_3007070((__int64)&v83) )
LABEL_55:
                  BUG();
                v39 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v10 + 592LL);
                if ( v39 == sub_2D56A50 )
                {
                  sub_2FE6CC0((__int64)&v85, v10, v65, v83, v84);
                  v40 = v59;
                  LOWORD(v40) = v86;
                  v41 = v87;
                  v42 = v65;
                  v59 = v40;
                }
                else
                {
                  v51 = v39(v10, v65, v83, v57);
                  v42 = v65;
                  v59 = v51;
                  v41 = v52;
                }
                v13 = sub_2FE98B0(v10, v42, (unsigned int)v59, v41);
              }
              goto LABEL_42;
            }
          }
          goto LABEL_41;
        }
        v86 = 0;
        LOWORD(v85) = 0;
        LOWORD(v81) = 0;
        sub_2FE8D10(v10, v63, (unsigned int)v79, v29, &v85, (unsigned int *)&v83, (unsigned __int16 *)&v81);
LABEL_46:
        v13 = (unsigned __int16)v81;
        goto LABEL_42;
      }
    }
LABEL_41:
    v13 = *(unsigned __int16 *)(v10 + 2 * v26 + 2852);
LABEL_42:
    v10 = *(_QWORD *)(a1 + 16);
    v12 = *a2;
    goto LABEL_5;
  }
  v9 = 0;
LABEL_13:
  if ( v7 != (__int64 *)v90 )
    _libc_free((unsigned __int64)v7);
  return v9;
}
