// Function: sub_F2AF70
// Address: 0xf2af70
//
__int64 __fastcall sub_F2AF70(_QWORD *a1, __int64 a2)
{
  unsigned __int8 *v4; // r15
  int v5; // eax
  int v6; // edx
  __int64 v7; // rcx
  __int64 result; // rax
  char v9; // al
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 *v15; // rbx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 v19; // r12
  __int64 v20; // rdx
  int v21; // edi
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rcx
  __int64 v25; // rdi
  __int64 v26; // rbx
  _QWORD *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 *v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdi
  _QWORD *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  bool v40; // zf
  __int64 v41; // rbx
  _BYTE *v42; // rdi
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 *v45; // r15
  const char *v46; // rax
  __int64 v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // r10
  __int64 v50; // rbx
  unsigned int **v51; // r15
  __int64 v52; // rax
  __int64 v53; // r15
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rcx
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 *v60; // rbx
  __int64 v61; // rsi
  __int64 v62; // rsi
  __int64 v63; // r12
  __int64 v64; // rax
  __int64 *v65; // rcx
  __int64 v66; // rsi
  __int64 v67; // rsi
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rbx
  __int64 i; // r15
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 *v74; // rcx
  __int64 v75; // r9
  __int64 v76; // rax
  __int64 *v77; // [rsp+8h] [rbp-D8h]
  __int64 *v78; // [rsp+10h] [rbp-D0h]
  __int64 v79; // [rsp+28h] [rbp-B8h]
  __int64 v80; // [rsp+28h] [rbp-B8h]
  __int64 v81; // [rsp+28h] [rbp-B8h]
  __int64 v82; // [rsp+28h] [rbp-B8h]
  __int64 v83; // [rsp+28h] [rbp-B8h]
  __int64 v84; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v85; // [rsp+40h] [rbp-A0h] BYREF
  __int64 *v86; // [rsp+48h] [rbp-98h] BYREF
  __int64 v87[4]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v88; // [rsp+70h] [rbp-70h]
  __int64 *v89; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v90[3]; // [rsp+88h] [rbp-58h] BYREF
  __int16 v91; // [rsp+A0h] [rbp-40h]

  v4 = *(unsigned __int8 **)(a2 - 96);
  v90[0] = &v84;
  v89 = 0;
  if ( *v4 != 59 )
  {
LABEL_2:
    if ( *v4 == 86 )
    {
      v89 = &v84;
      v90[0] = 0;
      v90[1] = &v85;
      v38 = *((_QWORD *)v4 + 2);
      if ( v38 )
      {
        if ( !*(_QWORD *)(v38 + 8) && *v4 > 0x1Cu )
        {
          v39 = *((_QWORD *)v4 + 1);
          if ( (unsigned int)*(unsigned __int8 *)(v39 + 8) - 17 <= 1 )
            v39 = **(_QWORD **)(v39 + 16);
          v40 = !sub_BCAC40(v39, 1);
          v5 = *v4;
          if ( v40 )
            goto LABEL_4;
          if ( (_BYTE)v5 == 57 )
          {
            if ( (v4[7] & 0x40) != 0 )
              v74 = (__int64 *)*((_QWORD *)v4 - 1);
            else
              v74 = (__int64 *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
            if ( !*v74 )
              goto LABEL_4;
            v75 = v74[4];
            *v89 = *v74;
            v76 = *(_QWORD *)(v75 + 16);
            if ( v76 && !*(_QWORD *)(v76 + 8) && (unsigned __int8)sub_996420(v90, 30, (unsigned __int8 *)v75) )
              goto LABEL_76;
          }
          else
          {
            if ( (_BYTE)v5 != 86 )
              goto LABEL_4;
            v41 = *((_QWORD *)v4 - 12);
            if ( *(_QWORD *)(v41 + 8) != *((_QWORD *)v4 + 1) )
              goto LABEL_4;
            v42 = (_BYTE *)*((_QWORD *)v4 - 4);
            if ( *v42 > 0x15u )
              goto LABEL_4;
            v43 = *((_QWORD *)v4 - 8);
            if ( sub_AC30F0((__int64)v42) )
            {
              *v89 = v41;
              v44 = *(_QWORD *)(v43 + 16);
              if ( v44 )
              {
                if ( !*(_QWORD *)(v44 + 8) && (unsigned __int8)sub_996420(v90, 30, (unsigned __int8 *)v43) )
                {
LABEL_76:
                  v45 = (__int64 *)a1[4];
                  v46 = sub_BD5D20(v84);
                  v87[0] = (__int64)"not.";
                  v47 = v84;
                  v88 = 1283;
                  v87[3] = v48;
                  v87[2] = (__int64)v46;
                  v80 = sub_AD62B0(*(_QWORD *)(v84 + 8));
                  v49 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v45[10] + 16LL))(
                          v45[10],
                          30,
                          v47,
                          v80);
                  if ( !v49 )
                  {
                    v91 = 257;
                    v82 = sub_B504D0(30, v47, v80, (__int64)&v89, 0, 0);
                    (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v45[11] + 16LL))(
                      v45[11],
                      v82,
                      v87,
                      v45[7],
                      v45[8]);
                    v70 = *v45;
                    v49 = v82;
                    for ( i = *v45 + 16LL * *((unsigned int *)v45 + 2); i != v70; v49 = v83 )
                    {
                      v72 = *(_QWORD *)(v70 + 8);
                      v73 = *(_DWORD *)v70;
                      v70 += 16;
                      v83 = v49;
                      sub_B99FD0(v49, v73, v72);
                    }
                  }
                  v50 = v85;
                  v51 = (unsigned int **)a1[4];
                  v81 = v49;
                  v91 = 257;
                  v52 = sub_AD62B0(*(_QWORD *)(v85 + 8));
                  v53 = sub_B36550(v51, v81, v52, v50, (__int64)&v89, 0);
                  sub_B4CC70(a2);
                  v54 = a1[23];
                  if ( v54 )
                    sub_FF0720(v54, *(_QWORD *)(a2 + 40));
                  result = a2;
                  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
                    v55 = *(_QWORD *)(a2 - 8);
                  else
                    v55 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
                  v15 = *(__int64 **)v55;
                  if ( *(_QWORD *)v55 )
                  {
                    v56 = *(_QWORD *)(v55 + 8);
                    **(_QWORD **)(v55 + 16) = v56;
                    if ( v56 )
                      *(_QWORD *)(v56 + 16) = *(_QWORD *)(v55 + 16);
                  }
                  *(_QWORD *)v55 = v53;
                  if ( v53 )
                  {
                    v57 = *(_QWORD *)(v53 + 16);
                    *(_QWORD *)(v55 + 8) = v57;
                    if ( v57 )
                      *(_QWORD *)(v57 + 16) = v55 + 8;
                    *(_QWORD *)(v55 + 16) = v53 + 16;
                    *(_QWORD *)(v53 + 16) = v55;
                  }
                  goto LABEL_26;
                }
              }
            }
          }
        }
      }
    }
    v5 = *v4;
LABEL_4:
    LOBYTE(v6) = v5;
    if ( (_BYTE)v5 == 17 )
    {
LABEL_59:
      v37 = (_QWORD *)*((_QWORD *)v4 + 3);
      if ( *((_DWORD *)v4 + 8) > 0x40u )
        v37 = (_QWORD *)*v37;
      sub_F26260((__int64)a1, *(_QWORD *)(a2 + 40), *(_QWORD *)(a2 - 32LL * (v37 == 0) - 32));
      return 0;
    }
    if ( *(_QWORD *)(a2 - 32) == *(_QWORD *)(a2 - 64) )
    {
      v58 = sub_AD6450(*((_QWORD *)v4 + 1));
      result = a2;
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v59 = *(_QWORD *)(a2 - 8);
      else
        v59 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v60 = *(__int64 **)v59;
      if ( *(_QWORD *)v59 )
      {
        v61 = *(_QWORD *)(v59 + 8);
        **(_QWORD **)(v59 + 16) = v61;
        if ( v61 )
          *(_QWORD *)(v61 + 16) = *(_QWORD *)(v59 + 16);
      }
      *(_QWORD *)v59 = v58;
      if ( v58 )
      {
        v62 = *(_QWORD *)(v58 + 16);
        *(_QWORD *)(v59 + 8) = v62;
        if ( v62 )
          *(_QWORD *)(v62 + 16) = v59 + 8;
        *(_QWORD *)(v59 + 16) = v58 + 16;
        *(_QWORD *)(v58 + 16) = v59;
      }
      if ( *(_BYTE *)v60 > 0x1Cu )
      {
        v63 = a1[5];
        v79 = result;
        v89 = v60;
        v19 = v63 + 2096;
        sub_F200C0(v19, (__int64 *)&v89);
        v20 = v60[2];
        result = v79;
        if ( v20 )
        {
          if ( !*(_QWORD *)(v20 + 8) )
            goto LABEL_29;
        }
      }
      return result;
    }
    v7 = *((_QWORD *)v4 + 2);
    if ( v7 && !*(_QWORD *)(v7 + 8) && (_BYTE)v5 == 83 )
    {
      v21 = sub_B53900((__int64)v4);
      switch ( v21 )
      {
        case 3:
        case 5:
        case 6:
        case 33:
        case 35:
        case 37:
        case 39:
        case 41:
          *((_WORD *)v4 + 1) = sub_B52870(v21) | *((_WORD *)v4 + 1) & 0xFFC0;
          sub_B4CC70(a2);
          v36 = a1[23];
          if ( v36 )
            sub_FF0720(v36, *(_QWORD *)(a2 + 40));
          sub_F15FC0(a1[5], (__int64)v4);
          result = a2;
          break;
        default:
          v6 = *v4;
          if ( (unsigned int)(v6 - 12) <= 1 )
            goto LABEL_8;
          if ( (_BYTE)v6 != 17 )
            goto LABEL_37;
          goto LABEL_59;
      }
      return result;
    }
    if ( (unsigned int)(v5 - 12) <= 1 )
    {
LABEL_8:
      sub_F26260((__int64)a1, *(_QWORD *)(a2 + 40), 0);
      return 0;
    }
LABEL_37:
    if ( (unsigned __int8)v6 <= 0x15u
      || (v22 = *(_QWORD *)(a2 - 32), v22 == *(_QWORD *)(a2 - 64))
      || (v23 = *((_QWORD *)v4 + 2)) == 0 )
    {
LABEL_88:
      sub_FFACA0(a1 + 25, a2);
      return 0;
    }
    while ( 1 )
    {
      v24 = *(_QWORD *)(a2 + 40);
      v25 = a1[10];
      v26 = *(_QWORD *)(v23 + 8);
      v87[1] = v22;
      v87[0] = v24;
      if ( (unsigned __int8)sub_B19ED0(v25, v87, v23) )
        break;
      v27 = *(_QWORD **)(a2 - 64);
      v28 = a1[10];
      v89 = *(__int64 **)(a2 + 40);
      v90[0] = v27;
      if ( !(unsigned __int8)sub_B19ED0(v28, (__int64 *)&v89, v23) )
        goto LABEL_41;
      v29 = sub_AD6450(*((_QWORD *)v4 + 1));
      v30 = *(__int64 **)v23;
      if ( *(_QWORD *)v23 )
      {
        v31 = *(_QWORD *)(v23 + 8);
        **(_QWORD **)(v23 + 16) = v31;
        if ( v31 )
          *(_QWORD *)(v31 + 16) = *(_QWORD *)(v23 + 16);
      }
      *(_QWORD *)v23 = v29;
      if ( v29 )
      {
        v32 = *(_QWORD *)(v29 + 16);
        *(_QWORD *)(v23 + 8) = v32;
        if ( v32 )
          *(_QWORD *)(v32 + 16) = v23 + 8;
        *(_QWORD *)(v23 + 16) = v29 + 16;
        *(_QWORD *)(v29 + 16) = v23;
      }
      v33 = a1[5];
      if ( *(_BYTE *)v30 > 0x1Cu )
      {
        v86 = v30;
        v77 = v30;
        sub_F200C0(v33 + 2096, (__int64 *)&v86);
        v34 = v33 + 2096;
        v35 = v77[2];
        if ( v35 )
        {
          if ( !*(_QWORD *)(v35 + 8) )
          {
            v86 = *(__int64 **)(v35 + 24);
            sub_F200C0(v34, (__int64 *)&v86);
          }
        }
LABEL_54:
        v33 = a1[5];
      }
LABEL_55:
      sub_F15FC0(v33, *(_QWORD *)(v23 + 24));
LABEL_41:
      if ( !v26 )
        goto LABEL_88;
      v22 = *(_QWORD *)(a2 - 32);
      v23 = v26;
    }
    v64 = sub_AD6400(*((_QWORD *)v4 + 1));
    v65 = *(__int64 **)v23;
    if ( *(_QWORD *)v23 )
    {
      v66 = *(_QWORD *)(v23 + 8);
      **(_QWORD **)(v23 + 16) = v66;
      if ( v66 )
        *(_QWORD *)(v66 + 16) = *(_QWORD *)(v23 + 16);
    }
    *(_QWORD *)v23 = v64;
    if ( v64 )
    {
      v67 = *(_QWORD *)(v64 + 16);
      *(_QWORD *)(v23 + 8) = v67;
      if ( v67 )
        *(_QWORD *)(v67 + 16) = v23 + 8;
      *(_QWORD *)(v23 + 16) = v64 + 16;
      *(_QWORD *)(v64 + 16) = v23;
    }
    v33 = a1[5];
    if ( *(_BYTE *)v65 <= 0x1Cu )
      goto LABEL_55;
    v89 = v65;
    v78 = v65;
    sub_F200C0(v33 + 2096, (__int64 *)&v89);
    v68 = v33 + 2096;
    v69 = v78[2];
    if ( v69 && !*(_QWORD *)(v69 + 8) )
    {
      v89 = *(__int64 **)(v69 + 24);
      sub_F200C0(v68, (__int64 *)&v89);
      v33 = a1[5];
      goto LABEL_55;
    }
    goto LABEL_54;
  }
  v9 = sub_995B10(&v89, *((_QWORD *)v4 - 8));
  v10 = *((_QWORD *)v4 - 4);
  if ( v9 && v10 )
  {
    *v90[0] = v10;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10(&v89, v10) )
      goto LABEL_2;
    v11 = *((_QWORD *)v4 - 8);
    if ( !v11 )
      goto LABEL_2;
    *v90[0] = v11;
  }
  if ( *(_BYTE *)v84 <= 0x15u )
    goto LABEL_2;
  sub_B4CC70(a2);
  v12 = a1[23];
  if ( v12 )
    sub_FF0720(v12, *(_QWORD *)(a2 + 40));
  v13 = v84;
  result = a2;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v14 = *(_QWORD *)(a2 - 8);
  else
    v14 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v15 = *(__int64 **)v14;
  if ( *(_QWORD *)v14 )
  {
    v16 = *(_QWORD *)(v14 + 8);
    **(_QWORD **)(v14 + 16) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(v14 + 16);
  }
  *(_QWORD *)v14 = v13;
  if ( v13 )
  {
    v17 = *(_QWORD *)(v13 + 16);
    *(_QWORD *)(v14 + 8) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = v14 + 8;
    *(_QWORD *)(v14 + 16) = v13 + 16;
    *(_QWORD *)(v13 + 16) = v14;
  }
LABEL_26:
  if ( *(_BYTE *)v15 > 0x1Cu )
  {
    v18 = a1[5];
    v79 = result;
    v89 = v15;
    v19 = v18 + 2096;
    sub_F200C0(v19, (__int64 *)&v89);
    v20 = v15[2];
    result = v79;
    if ( v20 )
    {
      if ( !*(_QWORD *)(v20 + 8) )
      {
LABEL_29:
        v89 = *(__int64 **)(v20 + 24);
        sub_F200C0(v19, (__int64 *)&v89);
        return v79;
      }
    }
  }
  return result;
}
