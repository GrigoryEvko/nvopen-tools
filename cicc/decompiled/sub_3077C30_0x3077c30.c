// Function: sub_3077C30
// Address: 0x3077c30
//
unsigned __int64 __fastcall sub_3077C30(
        __int64 a1,
        int a2,
        __int64 a3,
        int *a4,
        unsigned __int64 a5,
        int a6,
        __int64 a7)
{
  unsigned __int64 v10; // rbx
  int *v11; // r10
  int v13; // eax
  unsigned __int64 v14; // rbx
  int v15; // r13d
  __int64 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned int (__fastcall *v22)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v23; // rdx
  bool v24; // of
  unsigned __int64 v25; // rbx
  __int64 *v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdx
  unsigned int (__fastcall *v32)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v33; // rdx
  char v35; // al
  int i; // r14d
  __int64 *v37; // r9
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rdx
  unsigned int (__fastcall *v43)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v44; // rdx
  unsigned __int64 v45; // rbx
  __int64 *v46; // r9
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rdx
  unsigned int (__fastcall *v52)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
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
  __int64 *v75; // r9
  __int64 v76; // rax
  __int64 v77; // rdi
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // rdx
  unsigned int (__fastcall *v81)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v82; // rdx
  unsigned __int64 v83; // rbx
  __int64 *v84; // r9
  __int64 v85; // rax
  __int64 v86; // rdi
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // rdx
  unsigned int (__fastcall *v90)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v91; // rdx
  int v93; // [rsp+0h] [rbp-90h]
  __int64 *v95; // [rsp+0h] [rbp-90h]
  __int64 *v96; // [rsp+0h] [rbp-90h]
  int v97; // [rsp+0h] [rbp-90h]
  __int64 *v98; // [rsp+0h] [rbp-90h]
  __int64 *v99; // [rsp+0h] [rbp-90h]
  int *v100; // [rsp+0h] [rbp-90h]
  int v101; // [rsp+8h] [rbp-88h]
  int v102; // [rsp+8h] [rbp-88h]
  int v103[4]; // [rsp+Ch] [rbp-84h] BYREF
  char v104; // [rsp+1Fh] [rbp-71h] BYREF
  int v105; // [rsp+20h] [rbp-70h] BYREF
  int v106; // [rsp+24h] [rbp-6Ch] BYREF
  unsigned int **v107; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v108[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v109[10]; // [rsp+40h] [rbp-50h] BYREF

  v103[0] = a6;
  if ( !a5 )
    goto LABEL_4;
  v10 = *(int *)(a3 + 32);
  v11 = a4;
  if ( a2 == 6 )
  {
    if ( a5 > 2 && (v35 = sub_B4F0B0(a4, a5, v10, (int *)v109, v103), v11 = a4, v35) )
    {
      if ( (int)v10 >= LODWORD(v109[0]) + v103[0] )
      {
        a7 = sub_BCDA70(*(__int64 **)(a3 + 24), v109[0]);
LABEL_22:
        v14 = 0;
        v101 = *(_DWORD *)(a7 + 32);
        if ( v101 )
        {
          for ( i = 0; i != v101; ++i )
          {
            v37 = (__int64 *)a7;
            if ( (unsigned int)*(unsigned __int8 *)(a7 + 8) - 17 <= 1 )
              v37 = **(__int64 ***)(a7 + 16);
            v95 = v37;
            v38 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v37, 0);
            v39 = *(_QWORD *)(a1 + 24);
            v41 = v40;
            v42 = v38;
            v43 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v39 + 736LL);
            BYTE2(v109[0]) = 0;
            v44 = v43(v39, *v95, v42, v41, v109[0]);
            v24 = __OFADD__(v44, v14);
            v45 = v44 + v14;
            if ( v24 )
            {
              v45 = 0x8000000000000000LL;
              if ( v44 )
                v45 = 0x7FFFFFFFFFFFFFFFLL;
            }
            v46 = (__int64 *)a3;
            if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
              v46 = **(__int64 ***)(a3 + 16);
            v96 = v46;
            v47 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v46, 0);
            v48 = *(_QWORD *)(a1 + 24);
            v50 = v49;
            v51 = v47;
            v52 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v48 + 736LL);
            BYTE2(v109[0]) = 0;
            v53 = v52(v48, *v96, v51, v50, v109[0]);
            v24 = __OFADD__(v53, v45);
            v14 = v53 + v45;
            if ( v24 )
            {
              v14 = 0x8000000000000000LL;
              if ( v53 )
                v14 = 0x7FFFFFFFFFFFFFFFLL;
            }
          }
        }
        return v14;
      }
    }
    else
    {
      v100 = v11;
      if ( !(unsigned __int8)sub_B4EEA0(v11, a5, v10) && !(unsigned __int8)sub_B4EF10(v100, a5, v10) )
        sub_B4EF80((__int64)v100, a5, v10, v103);
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
    if ( !(unsigned __int8)sub_B4EDA0(a4, a5, v10) )
    {
      if ( !(unsigned __int8)sub_B4EE20(a4, a5, v10) )
      {
        v108[0] = a4;
        v109[1] = &v104;
        v109[2] = &v105;
        v109[0] = v108;
        v109[3] = &v106;
        v108[1] = a5;
        v105 = v10;
        v104 = 0;
        v106 = -1;
        v107 = (unsigned int **)v108;
        if ( !sub_3069010(&v107, a5, (__int64)&v106, v54, v55, v56, (__int64)v108, &v104, &v105, (unsigned int *)&v106) )
        {
          if ( (unsigned __int8)sub_B4EFF0(a4, a5, v10, v103) && a5 + v103[0] <= v10 )
          {
            a7 = sub_BCDA70(*(__int64 **)(a3 + 24), a5);
LABEL_46:
            v14 = 0;
            v102 = *(_DWORD *)(a7 + 32);
            if ( v102 )
            {
              for ( j = 0; j != v102; ++j )
              {
                v75 = (__int64 *)a3;
                if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
                  v75 = **(__int64 ***)(a3 + 16);
                v98 = v75;
                v76 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v75, 0);
                v77 = *(_QWORD *)(a1 + 24);
                v79 = v78;
                v80 = v76;
                v81 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v77 + 736LL);
                BYTE2(v108[0]) = 0;
                v82 = v81(v77, *v98, v80, v79, v108[0]);
                v24 = __OFADD__(v82, v14);
                v83 = v82 + v14;
                if ( v24 )
                {
                  v83 = 0x8000000000000000LL;
                  if ( v82 )
                    v83 = 0x7FFFFFFFFFFFFFFFLL;
                }
                v84 = (__int64 *)a7;
                if ( (unsigned int)*(unsigned __int8 *)(a7 + 8) - 17 <= 1 )
                  v84 = **(__int64 ***)(a7 + 16);
                v99 = v84;
                v85 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v84, 0);
                v86 = *(_QWORD *)(a1 + 24);
                v88 = v87;
                v89 = v85;
                v90 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v86 + 736LL);
                BYTE2(v108[0]) = 0;
                v91 = v90(v86, *v99, v89, v88, v108[0]);
                v24 = __OFADD__(v91, v83);
                v14 = v91 + v83;
                if ( v24 )
                {
                  v14 = 0x8000000000000000LL;
                  if ( v91 )
                    v14 = 0x7FFFFFFFFFFFFFFFLL;
                }
              }
            }
            return v14;
          }
          goto LABEL_6;
        }
        v103[0] = v106;
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
      BYTE2(v108[0]) = 0;
      v64 = v63(v59, *v57, v62, v61, v108[0]);
      v97 = *(_DWORD *)(a3 + 32);
      if ( v97 > 0 )
      {
        for ( k = 0; k != v97; ++k )
        {
          v66 = (__int64 *)a3;
          if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
            v66 = **(__int64 ***)(a3 + 16);
          v67 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v66, 0);
          v68 = *(_QWORD *)(a1 + 24);
          v70 = v69;
          v71 = v67;
          v72 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v68 + 736LL);
          BYTE2(v108[0]) = 0;
          v73 = v72(v68, *v66, v71, v70, v108[0]);
          v24 = __OFADD__(v73, v64);
          v64 += v73;
          if ( v24 )
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
  v13 = *(unsigned __int8 *)(a3 + 8);
  if ( (_BYTE)v13 == 17 )
  {
    v93 = *(_DWORD *)(a3 + 32);
    v14 = 0;
    if ( v93 > 0 )
    {
      v15 = 0;
      while ( 1 )
      {
        v16 = (__int64 *)a3;
        if ( (unsigned int)(v13 - 17) <= 1 )
          v16 = **(__int64 ***)(a3 + 16);
        v17 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v16, 0);
        v18 = *(_QWORD *)(a1 + 24);
        v20 = v19;
        v21 = v17;
        v22 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v18 + 736LL);
        BYTE2(v108[0]) = 0;
        v23 = v22(v18, *v16, v21, v20, v108[0]);
        v24 = __OFADD__(v23, v14);
        v25 = v23 + v14;
        if ( v24 )
        {
          v25 = 0x8000000000000000LL;
          if ( v23 )
            v25 = 0x7FFFFFFFFFFFFFFFLL;
        }
        v26 = (__int64 *)a3;
        if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
          v26 = **(__int64 ***)(a3 + 16);
        v27 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v26, 0);
        v28 = *(_QWORD *)(a1 + 24);
        v30 = v29;
        v31 = v27;
        v32 = *(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v28 + 736LL);
        BYTE2(v108[0]) = 0;
        v33 = v32(v28, *v26, v31, v30, v108[0]);
        v24 = __OFADD__(v33, v25);
        v14 = v33 + v25;
        if ( v24 )
        {
          v14 = 0x8000000000000000LL;
          if ( v33 )
            v14 = 0x7FFFFFFFFFFFFFFFLL;
        }
        if ( v93 == ++v15 )
          break;
        v13 = *(unsigned __int8 *)(a3 + 8);
      }
    }
    return v14;
  }
  return 0;
}
