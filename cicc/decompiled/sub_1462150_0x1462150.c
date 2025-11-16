// Function: sub_1462150
// Address: 0x1462150
//
__int64 __fastcall sub_1462150(
        _QWORD *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7)
{
  __int64 v8; // r13
  int v9; // edx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rbx
  unsigned int v12; // r8d
  _QWORD *v16; // rsi
  _QWORD *v17; // rdi
  __int64 *v18; // r8
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // r11
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  __int64 *v24; // rbx
  __int64 **v25; // rsi
  __int64 *v26; // rax
  __int64 *v27; // r11
  __int64 *v28; // rdi
  __int64 *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // r11
  __int64 *v33; // rbx
  __int64 *v34; // rax
  __int64 *v35; // rdi
  _BYTE *v36; // rax
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // eax
  int v41; // esi
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r13
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rbx
  unsigned __int64 v47; // r12
  unsigned int v48; // eax
  unsigned __int64 v49; // r11
  __int64 v50; // r13
  int v51; // ebx
  _QWORD *v52; // [rsp+8h] [rbp-98h]
  __int64 v53; // [rsp+8h] [rbp-98h]
  __int64 v54; // [rsp+10h] [rbp-90h]
  __int64 v55; // [rsp+10h] [rbp-90h]
  __int64 v56; // [rsp+18h] [rbp-88h]
  __int64 *v57; // [rsp+18h] [rbp-88h]
  _QWORD *v58; // [rsp+20h] [rbp-80h]
  _QWORD *v59; // [rsp+20h] [rbp-80h]
  __int64 *v60; // [rsp+28h] [rbp-78h]
  __int64 v61; // [rsp+28h] [rbp-78h]
  __int64 v62; // [rsp+28h] [rbp-78h]
  __int64 *v63; // [rsp+30h] [rbp-70h]
  int v64; // [rsp+30h] [rbp-70h]
  __int64 *v65; // [rsp+38h] [rbp-68h]
  __int64 *v66; // [rsp+38h] [rbp-68h]
  int v67; // [rsp+38h] [rbp-68h]
  unsigned __int64 v68; // [rsp+38h] [rbp-68h]
  _QWORD *v69; // [rsp+40h] [rbp-60h]
  __int64 *v70; // [rsp+40h] [rbp-60h]
  unsigned int v71; // [rsp+40h] [rbp-60h]
  int v72; // [rsp+40h] [rbp-60h]
  __int64 v73; // [rsp+40h] [rbp-60h]
  __int64 v74; // [rsp+40h] [rbp-60h]
  unsigned __int64 v75; // [rsp+48h] [rbp-58h] BYREF
  __int64 v76; // [rsp+50h] [rbp-50h] BYREF
  __int64 v77; // [rsp+58h] [rbp-48h]
  unsigned __int64 v78; // [rsp+60h] [rbp-40h]

  v75 = a5;
  if ( (__int64 *)a5 == a4 )
    return 0;
  v8 = *((unsigned __int16 *)a4 + 12);
  v9 = *(unsigned __int16 *)(a5 + 24);
  v10 = (unsigned __int64)a4;
  v11 = a5;
  if ( (_WORD)v8 != (_WORD)v9 )
    return (unsigned int)(unsigned __int16)v8 - v9;
  if ( a7 > dword_4F9B0A0 )
    return 0;
  v16 = a1 + 1;
  v17 = (_QWORD *)a1[2];
  v18 = &v76;
  v77 = 1;
  v76 = (__int64)&v76;
  v19 = v17;
  v78 = (unsigned __int64)a4;
  if ( v17 )
  {
    v20 = v16;
    do
    {
      while ( 1 )
      {
        v21 = v19[2];
        a4 = (__int64 *)v19[3];
        if ( v19[6] >= v10 )
          break;
        v19 = (_QWORD *)v19[3];
        if ( !a4 )
          goto LABEL_11;
      }
      v20 = v19;
      v19 = (_QWORD *)v19[2];
    }
    while ( v21 );
LABEL_11:
    if ( v16 != v20 && v20[6] <= v10 )
    {
      a4 = v20 + 4;
      if ( (v20[5] & 1) == 0 )
      {
        a4 = (__int64 *)v20[4];
        if ( (a4[1] & 1) == 0 )
        {
          v32 = (__int64 *)*a4;
          if ( (*(_BYTE *)(*a4 + 8) & 1) != 0 )
          {
            v20[4] = v32;
            a4 = v32;
            v17 = (_QWORD *)a1[2];
          }
          else
          {
            v33 = (__int64 *)*v32;
            if ( (*(_BYTE *)(*v32 + 8) & 1) == 0 )
            {
              v66 = (__int64 *)*v33;
              if ( (*(_BYTE *)(*v33 + 8) & 1) != 0 )
              {
                v33 = (__int64 *)*v33;
              }
              else
              {
                v34 = *(_BYTE **)*v33;
                v70 = (__int64 *)v34;
                if ( (v34[8] & 1) == 0 )
                {
                  v35 = *(_BYTE **)v34;
                  if ( (*(_BYTE *)(*(_QWORD *)v34 + 8LL) & 1) == 0 )
                  {
                    v53 = a6;
                    v55 = a3;
                    v57 = (__int64 *)*a4;
                    v59 = v20;
                    v63 = (__int64 *)v20[4];
                    v36 = sub_145F440((__int64 *)v35);
                    v18 = &v76;
                    v20 = v59;
                    v35 = v36;
                    a6 = v53;
                    a3 = v55;
                    v32 = v57;
                    a4 = v63;
                    *v70 = (__int64)v36;
                  }
                  v70 = (__int64 *)v35;
                  *v66 = (__int64)v35;
                }
                *v33 = (__int64)v70;
                v33 = v70;
              }
              *v32 = (__int64)v33;
            }
            *a4 = (__int64)v33;
            v20[4] = v33;
            if ( !v33 )
            {
LABEL_26:
              v11 = v75;
              goto LABEL_27;
            }
            a4 = v33;
            v17 = (_QWORD *)a1[2];
            v11 = v75;
          }
        }
      }
      v76 = (__int64)&v76;
      v22 = v17;
      v77 = 1;
      v78 = v11;
      if ( !v17 )
        goto LABEL_27;
      v23 = v16;
      do
      {
        v18 = (__int64 *)v22[2];
        if ( v11 > v22[6] )
        {
          v22 = (_QWORD *)v22[3];
        }
        else
        {
          v23 = v22;
          v22 = (_QWORD *)v22[2];
        }
      }
      while ( v22 );
      if ( v16 == v23 || v11 < v23[6] )
        goto LABEL_27;
      v24 = v23 + 4;
      if ( (v23[5] & 1) == 0 )
      {
        v24 = (__int64 *)v23[4];
        if ( (v24[1] & 1) == 0 )
        {
          v25 = (__int64 **)*v24;
          if ( (*(_BYTE *)(*v24 + 8) & 1) != 0 )
          {
            v24 = (__int64 *)*v24;
          }
          else
          {
            v18 = *v25;
            if ( ((*v25)[1] & 1) == 0 )
            {
              v26 = (__int64 *)*v18;
              v69 = (_QWORD *)*v18;
              if ( (*(_BYTE *)(*v18 + 8) & 1) != 0 )
              {
                v18 = (__int64 *)*v18;
              }
              else
              {
                v27 = (__int64 *)*v26;
                if ( (*(_BYTE *)(*v26 + 8) & 1) == 0 )
                {
                  v28 = (__int64 *)*v27;
                  v52 = (_QWORD *)*v26;
                  if ( (*(_BYTE *)(*v27 + 8) & 1) == 0 )
                  {
                    v54 = a6;
                    v56 = a3;
                    v58 = v23;
                    v60 = *v25;
                    v65 = a4;
                    v29 = (__int64 *)sub_145F440(v28);
                    a6 = v54;
                    a3 = v56;
                    v23 = v58;
                    v28 = v29;
                    v18 = v60;
                    *v52 = v29;
                    a4 = v65;
                  }
                  v27 = v28;
                  *v69 = v28;
                }
                *v18 = (__int64)v27;
                v18 = v27;
              }
              *v25 = v18;
            }
            *v24 = (__int64)v18;
            v24 = v18;
          }
          v23[4] = v24;
        }
      }
      if ( a4 == v24 )
        return 0;
      goto LABEL_26;
    }
  }
LABEL_27:
  switch ( v8 )
  {
    case 0LL:
      v38 = *(_QWORD *)(v10 + 32);
      v39 = *(_QWORD *)(v11 + 32);
      v40 = *(_DWORD *)(v38 + 32);
      v41 = *(_DWORD *)(v39 + 32);
      if ( v40 == v41 )
      {
        if ( (int)sub_16A9900(v38 + 24, v39 + 24) >= 0 )
          return 1;
        return (unsigned int)-1;
      }
      return (unsigned int)(v40 - v41);
    case 1LL:
    case 2LL:
    case 3LL:
      v37 = sub_1462150((_DWORD)a1, (_DWORD)a2, a3, *(_QWORD *)(v10 + 32), *(_QWORD *)(v11 + 32), a6, a7 + 1);
      goto LABEL_55;
    case 4LL:
    case 5LL:
    case 8LL:
    case 9LL:
      v30 = *(_QWORD *)(v10 + 40);
      v31 = *(_QWORD *)(v11 + 40);
      if ( (_DWORD)v30 != (_DWORD)v31 )
        return (unsigned int)(v30 - v31);
      if ( (*(_WORD *)(v10 + 26) & 7) != (*(_WORD *)(v11 + 26) & 7) )
        return (*(_WORD *)(v10 + 26) & 7) - (*(_WORD *)(v11 + 26) & 7u);
      if ( !(_DWORD)v30 )
        goto LABEL_80;
      v64 = a3;
      v44 = 0;
      v61 = 8LL * (unsigned int)v30;
      v45 = v11;
      v46 = v10;
      v47 = v45;
      break;
    case 6LL:
      v67 = a6;
      v72 = a3;
      v12 = sub_1462150((_DWORD)a1, (_DWORD)a2, a3, *(_QWORD *)(v10 + 32), *(_QWORD *)(v11 + 32), a6, a7 + 1);
      if ( !v12 )
      {
        v37 = sub_1462150((_DWORD)a1, (_DWORD)a2, v72, *(_QWORD *)(v10 + 40), *(_QWORD *)(v11 + 40), v67, a7 + 1);
LABEL_55:
        v12 = v37;
        if ( !v37 )
          goto LABEL_56;
      }
      return v12;
    case 7LL:
      v42 = *(_QWORD *)(v10 + 48);
      v43 = *(_QWORD *)(v11 + 48);
      if ( v42 != v43 )
      {
        if ( (unsigned __int8)sub_15CC8F0(a6, **(_QWORD **)(v42 + 32), **(_QWORD **)(v43 + 32), a4, v18) )
          return 1;
        else
          return (unsigned int)-1;
      }
      v30 = *(_QWORD *)(v10 + 40);
      v31 = *(_QWORD *)(v11 + 40);
      if ( (_DWORD)v30 != (_DWORD)v31 )
        return (unsigned int)(v30 - v31);
      if ( (*(_WORD *)(v10 + 26) & 7) != (*(_WORD *)(v11 + 26) & 7) )
        return (*(_WORD *)(v10 + 26) & 7) - (*(_WORD *)(v11 + 26) & 7u);
      if ( (_DWORD)v30 )
      {
        v49 = v11;
        v50 = 0;
        v51 = a3;
        v62 = 8LL * (unsigned int)v30;
        do
        {
          v74 = a6;
          v68 = v49;
          v12 = sub_1462150(
                  (_DWORD)a1,
                  (_DWORD)a2,
                  v51,
                  *(_QWORD *)(*(_QWORD *)(v10 + 32) + v50),
                  *(_QWORD *)(*(_QWORD *)(v49 + 32) + v50),
                  a6,
                  a7 + 1);
          if ( v12 )
            return v12;
          v50 += 8;
          a6 = v74;
          v49 = v68;
        }
        while ( v50 != v62 );
      }
      goto LABEL_80;
    case 10LL:
      if ( !v11 )
        BUG();
      v12 = sub_14616B0(a2, a3, *(_QWORD *)(v10 - 8), *(_QWORD *)(v11 - 8), a7 + 1);
      if ( !v12 )
      {
LABEL_56:
        v71 = v12;
        sub_1461F50(a1, v10, (__int64 *)&v75);
        return v71;
      }
      return v12;
  }
  do
  {
    v73 = a6;
    v48 = sub_1462150(
            (_DWORD)a1,
            (_DWORD)a2,
            v64,
            *(_QWORD *)(*(_QWORD *)(v46 + 32) + v44),
            *(_QWORD *)(*(_QWORD *)(v47 + 32) + v44),
            a6,
            a7 + 1);
    if ( v48 )
      return v48;
    v44 += 8;
    a6 = v73;
  }
  while ( v61 != v44 );
  v10 = v46;
LABEL_80:
  sub_1461F50(a1, v10, (__int64 *)&v75);
  return 0;
}
