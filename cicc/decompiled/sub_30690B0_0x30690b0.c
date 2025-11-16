// Function: sub_30690B0
// Address: 0x30690b0
//
unsigned __int64 __fastcall sub_30690B0(
        __int64 a1,
        int a2,
        __int64 a3,
        int *a4,
        unsigned __int64 a5,
        __int64 a6,
        signed int a7,
        __int64 a8)
{
  __int64 v10; // r13
  unsigned __int64 v11; // rbx
  int *v12; // r9
  int v14; // eax
  unsigned __int64 v15; // rbx
  int v16; // r13d
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned int (__fastcall *v23)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v24; // rdx
  bool v25; // of
  unsigned __int64 v26; // rbx
  __int64 *v27; // r14
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rdx
  unsigned int (__fastcall *v33)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v34; // rdx
  char v36; // al
  int i; // r14d
  __int64 *v38; // r10
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rdx
  unsigned int (__fastcall *v44)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v45; // rdx
  unsigned __int64 v46; // rbx
  __int64 *v47; // r10
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rdx
  unsigned int (__fastcall *v53)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 *v57; // rbx
  __int64 v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // rdx
  unsigned int (__fastcall *v63)(__int64, __int64, __int64, __int64, _QWORD); // rax
  unsigned __int64 v64; // rbx
  int k; // r14d
  __int64 *v66; // r13
  __int64 v67; // rax
  __int64 v68; // rdi
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // rdx
  unsigned int (__fastcall *v72)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v73; // rdx
  int j; // r14d
  __int64 *v75; // r10
  __int64 v76; // rax
  __int64 v77; // rdi
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // rdx
  unsigned int (__fastcall *v81)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v82; // rdx
  unsigned __int64 v83; // rbx
  __int64 *v84; // r10
  __int64 v85; // rax
  __int64 v86; // rdi
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // rdx
  unsigned int (__fastcall *v90)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v91; // rdx
  int v92; // [rsp+4h] [rbp-8Ch]
  int v93; // [rsp+4h] [rbp-8Ch]
  int v95; // [rsp+8h] [rbp-88h]
  __int64 *v97; // [rsp+8h] [rbp-88h]
  __int64 *v98; // [rsp+8h] [rbp-88h]
  int v99; // [rsp+8h] [rbp-88h]
  __int64 *v100; // [rsp+8h] [rbp-88h]
  __int64 *v101; // [rsp+8h] [rbp-88h]
  int *v102; // [rsp+8h] [rbp-88h]
  char v103; // [rsp+1Fh] [rbp-71h] BYREF
  int v104; // [rsp+20h] [rbp-70h] BYREF
  signed int v105; // [rsp+24h] [rbp-6Ch] BYREF
  unsigned int **v106; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v107[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v108[10]; // [rsp+40h] [rbp-50h] BYREF

  v10 = a8;
  if ( !a5 )
    goto LABEL_4;
  v11 = *(int *)(a3 + 32);
  v12 = a4;
  if ( a2 == 6 )
  {
    if ( a5 > 2 && (v36 = sub_B4F0B0(a4, a5, v11, (int *)v108, &a7), v12 = a4, v36) )
    {
      if ( (int)v11 >= LODWORD(v108[0]) + a7 )
      {
        v10 = sub_BCDA70(*(__int64 **)(a3 + 24), v108[0]);
LABEL_22:
        v15 = 0;
        v92 = *(_DWORD *)(v10 + 32);
        if ( v92 )
        {
          for ( i = 0; i != v92; ++i )
          {
            v38 = (__int64 *)v10;
            if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
              v38 = **(__int64 ***)(v10 + 16);
            v97 = v38;
            v39 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v38, 0);
            v40 = *(_QWORD *)(a1 + 24);
            v42 = v41;
            v43 = v39;
            v44 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v40 + 736LL);
            BYTE2(v108[0]) = 0;
            v45 = v44(v40, *v97, v43, v42, v108[0]);
            v25 = __OFADD__(v45, v15);
            v46 = v45 + v15;
            if ( v25 )
            {
              v46 = 0x8000000000000000LL;
              if ( v45 )
                v46 = 0x7FFFFFFFFFFFFFFFLL;
            }
            v47 = (__int64 *)a3;
            if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
              v47 = **(__int64 ***)(a3 + 16);
            v98 = v47;
            v48 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v47, 0);
            v49 = *(_QWORD *)(a1 + 24);
            v51 = v50;
            v52 = v48;
            v53 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v49 + 736LL);
            BYTE2(v108[0]) = 0;
            v54 = v53(v49, *v98, v52, v51, v108[0]);
            v25 = __OFADD__(v54, v46);
            v15 = v54 + v46;
            if ( v25 )
            {
              v15 = 0x8000000000000000LL;
              if ( v54 )
                v15 = 0x7FFFFFFFFFFFFFFFLL;
            }
          }
        }
        return v15;
      }
    }
    else
    {
      v102 = v12;
      if ( !(unsigned __int8)sub_B4EEA0(v12, a5, v11) && !(unsigned __int8)sub_B4EF10(v102, a5, v11) )
        sub_B4EF80((__int64)v102, a5, v11, &a7);
    }
  }
  else
  {
    if ( a2 != 7 )
    {
LABEL_4:
      switch ( a2 )
      {
        case 0:
          goto LABEL_35;
        case 1:
        case 2:
        case 3:
        case 6:
        case 7:
        case 8:
          goto LABEL_6;
        case 4:
          goto LABEL_22;
        case 5:
          goto LABEL_46;
        default:
          BUG();
      }
    }
    if ( !(unsigned __int8)sub_B4EDA0(a4, a5, v11) )
    {
      if ( !(unsigned __int8)sub_B4EE20(a4, a5, v11) )
      {
        v107[0] = a4;
        v108[1] = &v103;
        v108[2] = &v104;
        v108[0] = v107;
        v108[3] = &v105;
        v107[1] = a5;
        v104 = v11;
        v103 = 0;
        v105 = -1;
        v106 = (unsigned int **)v107;
        if ( !sub_3069010(
                &v106,
                a5,
                (__int64)&v105,
                v55,
                v56,
                (__int64)a4,
                (__int64)v107,
                &v103,
                &v104,
                (unsigned int *)&v105) )
        {
          if ( (unsigned __int8)sub_B4EFF0(a4, a5, v11, &a7) && a5 + a7 <= v11 )
          {
            v10 = sub_BCDA70(*(__int64 **)(a3 + 24), a5);
LABEL_46:
            v15 = 0;
            v93 = *(_DWORD *)(v10 + 32);
            if ( v93 )
            {
              for ( j = 0; j != v93; ++j )
              {
                v75 = (__int64 *)a3;
                if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
                  v75 = **(__int64 ***)(a3 + 16);
                v100 = v75;
                v76 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v75, 0);
                v77 = *(_QWORD *)(a1 + 24);
                v79 = v78;
                v80 = v76;
                v81 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v77 + 736LL);
                BYTE2(v107[0]) = 0;
                v82 = v81(v77, *v100, v80, v79, v107[0]);
                v25 = __OFADD__(v82, v15);
                v83 = v82 + v15;
                if ( v25 )
                {
                  v83 = 0x8000000000000000LL;
                  if ( v82 )
                    v83 = 0x7FFFFFFFFFFFFFFFLL;
                }
                v84 = (__int64 *)v10;
                if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
                  v84 = **(__int64 ***)(v10 + 16);
                v101 = v84;
                v85 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v84, 0);
                v86 = *(_QWORD *)(a1 + 24);
                v88 = v87;
                v89 = v85;
                v90 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v86 + 736LL);
                BYTE2(v107[0]) = 0;
                v91 = v90(v86, *v101, v89, v88, v107[0]);
                v25 = __OFADD__(v91, v83);
                v15 = v91 + v83;
                if ( v25 )
                {
                  v15 = 0x8000000000000000LL;
                  if ( v91 )
                    v15 = 0x7FFFFFFFFFFFFFFFLL;
                }
              }
            }
            return v15;
          }
          goto LABEL_6;
        }
        a7 = v105;
      }
LABEL_35:
      if ( *(_BYTE *)(a3 + 8) != 17 )
        return 0;
      v57 = **(__int64 ***)(a3 + 16);
      v58 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v57, 0);
      v59 = *(_QWORD *)(a1 + 24);
      v61 = v60;
      v62 = v58;
      v63 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v59 + 736LL);
      BYTE2(v107[0]) = 0;
      v64 = v63(v59, *v57, v62, v61, v107[0]);
      v99 = *(_DWORD *)(a3 + 32);
      if ( v99 > 0 )
      {
        for ( k = 0; k != v99; ++k )
        {
          v66 = (__int64 *)a3;
          if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
            v66 = **(__int64 ***)(a3 + 16);
          v67 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v66, 0);
          v68 = *(_QWORD *)(a1 + 24);
          v70 = v69;
          v71 = v67;
          v72 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v68 + 736LL);
          BYTE2(v107[0]) = 0;
          v73 = v72(v68, *v66, v71, v70, v107[0]);
          v25 = __OFADD__(v73, v64);
          v64 += v73;
          if ( v25 )
          {
            v64 = 0x8000000000000000LL;
            if ( v73 )
              v64 = 0x7FFFFFFFFFFFFFFFLL;
          }
        }
      }
      return v64;
    }
  }
LABEL_6:
  v14 = *(unsigned __int8 *)(a3 + 8);
  if ( (_BYTE)v14 == 17 )
  {
    v95 = *(_DWORD *)(a3 + 32);
    v15 = 0;
    if ( v95 > 0 )
    {
      v16 = 0;
      while ( 1 )
      {
        v17 = (__int64 *)a3;
        if ( (unsigned int)(v14 - 17) <= 1 )
          v17 = **(__int64 ***)(a3 + 16);
        v18 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v17, 0);
        v19 = *(_QWORD *)(a1 + 24);
        v21 = v20;
        v22 = v18;
        v23 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v19 + 736LL);
        BYTE2(v107[0]) = 0;
        v24 = v23(v19, *v17, v22, v21, v107[0]);
        v25 = __OFADD__(v24, v15);
        v26 = v24 + v15;
        if ( v25 )
        {
          v26 = 0x8000000000000000LL;
          if ( v24 )
            v26 = 0x7FFFFFFFFFFFFFFFLL;
        }
        v27 = (__int64 *)a3;
        if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
          v27 = **(__int64 ***)(a3 + 16);
        v28 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v27, 0);
        v29 = *(_QWORD *)(a1 + 24);
        v31 = v30;
        v32 = v28;
        v33 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v29 + 736LL);
        BYTE2(v107[0]) = 0;
        v34 = v33(v29, *v27, v32, v31, v107[0]);
        v25 = __OFADD__(v34, v26);
        v15 = v34 + v26;
        if ( v25 )
        {
          v15 = 0x8000000000000000LL;
          if ( v34 )
            v15 = 0x7FFFFFFFFFFFFFFFLL;
        }
        if ( v95 == ++v16 )
          break;
        v14 = *(unsigned __int8 *)(a3 + 8);
      }
    }
    return v15;
  }
  return 0;
}
