// Function: sub_111E610
// Address: 0x111e610
//
__int64 *__fastcall sub_111E610(__int64 a1, __int64 a2)
{
  __int16 v3; // bx
  __int64 *v4; // r15
  __int64 *v5; // r12
  int v6; // ebx
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rdx
  int v11; // ecx
  int v12; // eax
  _QWORD *v13; // rdi
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v17; // r13
  __int64 v18; // rdx
  int v19; // ecx
  int v20; // eax
  _QWORD *v21; // rdi
  __int64 *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  _BYTE *v25; // rax
  __int64 **v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  _BYTE *v30; // rdx
  __int64 **v31; // r13
  _BYTE *v32; // rsi
  __int64 *v33; // rsi
  __int64 v34; // rdx
  unsigned int v35; // r13d
  bool v36; // al
  __int64 v37; // rax
  __int64 **v38; // rax
  __int64 **v39; // rax
  __int64 v40; // r13
  _BYTE *v41; // rax
  unsigned int v42; // r13d
  _BYTE *v43; // rdx
  __int64 **v44; // r13
  unsigned int v45; // r13d
  __int64 v46; // rax
  char v47; // al
  char v48; // al
  char v49; // al
  bool v50; // zf
  char v51; // [rsp+4h] [rbp-8Ch]
  int v52; // [rsp+4h] [rbp-8Ch]
  bool v53; // [rsp+8h] [rbp-88h]
  int v54; // [rsp+8h] [rbp-88h]
  __int64 v55; // [rsp+8h] [rbp-88h]
  __int64 v56; // [rsp+8h] [rbp-88h]
  __int64 *v57; // [rsp+18h] [rbp-78h] BYREF
  __int64 v58; // [rsp+20h] [rbp-70h]
  __int64 v59; // [rsp+28h] [rbp-68h]
  __int64 *v60; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v61; // [rsp+38h] [rbp-58h] BYREF
  __int64 **v62; // [rsp+40h] [rbp-50h]
  __int16 v63; // [rsp+50h] [rbp-40h]

  v3 = *(_WORD *)(a1 + 2);
  v57 = 0;
  v4 = *(__int64 **)(a1 - 64);
  v5 = *(__int64 **)(a1 - 32);
  v6 = v3 & 0x3F;
  if ( (unsigned int)(v6 - 32) <= 1 )
  {
    v61 = 0;
    v60 = (__int64 *)&v57;
    v62 = &v57;
    v27 = v4[2];
    if ( !v27 || *(_QWORD *)(v27 + 8) || *(_BYTE *)v4 != 57 )
      goto LABEL_29;
    v32 = (_BYTE *)*(v4 - 8);
    if ( *v32 == 42 )
    {
      v47 = sub_1111CE0(&v60, (__int64)v32);
      v33 = (__int64 *)*(v4 - 4);
      if ( v47 && v33 == *v62 )
      {
LABEL_52:
        if ( *(_BYTE *)v5 == 17 )
        {
          v35 = *((_DWORD *)v5 + 8);
          if ( v35 <= 0x40 )
            v36 = v5[3] == 0;
          else
            v36 = v35 == (unsigned int)sub_C444A0((__int64)(v5 + 3));
LABEL_55:
          if ( v36 )
            goto LABEL_30;
          goto LABEL_29;
        }
        v40 = v5[1];
        if ( (unsigned int)*(unsigned __int8 *)(v40 + 8) - 17 <= 1 && *(_BYTE *)v5 <= 0x15u )
        {
          v41 = sub_AD7630((__int64)v5, 0, v34);
          if ( v41 && *v41 == 17 )
          {
            v42 = *((_DWORD *)v41 + 8);
            if ( v42 > 0x40 )
            {
              v36 = v42 == (unsigned int)sub_C444A0((__int64)(v41 + 24));
              goto LABEL_55;
            }
            if ( !*((_QWORD *)v41 + 3) )
              goto LABEL_30;
          }
          else if ( *(_BYTE *)(v40 + 8) == 17 )
          {
            v54 = *(_DWORD *)(v40 + 32);
            if ( v54 )
            {
              v51 = 0;
              v45 = 0;
              while ( 1 )
              {
                v46 = sub_AD69F0((unsigned __int8 *)v5, v45);
                if ( !v46 )
                  break;
                if ( *(_BYTE *)v46 != 13 )
                {
                  if ( *(_BYTE *)v46 != 17 )
                    break;
                  if ( *(_DWORD *)(v46 + 32) <= 0x40u )
                  {
                    if ( *(_QWORD *)(v46 + 24) )
                      break;
                  }
                  else
                  {
                    v52 = *(_DWORD *)(v46 + 32);
                    if ( v52 != (unsigned int)sub_C444A0(v46 + 24) )
                      break;
                  }
                  v51 = 1;
                }
                if ( v54 == ++v45 )
                {
                  if ( v51 )
                    goto LABEL_30;
                  break;
                }
              }
            }
          }
        }
LABEL_29:
        v57 = 0;
LABEL_30:
        v60 = 0;
        v61 = v5;
        v62 = (__int64 **)v5;
        v28 = v4[2];
        if ( v28 && !*(_QWORD *)(v28 + 8) && *(_BYTE *)v4 == 57 )
        {
          v30 = (_BYTE *)*(v4 - 8);
          if ( *v30 == 44 )
          {
            v55 = *(v4 - 8);
            v48 = sub_10081F0(&v60, *((_QWORD *)v30 - 8));
            v31 = (__int64 **)*(v4 - 4);
            if ( v48 && *(__int64 **)(v55 - 32) == v61 && v62 == v31 )
              goto LABEL_44;
          }
          else
          {
            v31 = (__int64 **)*(v4 - 4);
          }
          if ( *(_BYTE *)v31 == 44
            && (unsigned __int8)sub_10081F0(&v60, (__int64)*(v31 - 8))
            && *(v31 - 4) == v61
            && (__int64 **)*(v4 - 8) == v62 )
          {
LABEL_44:
            v57 = v5;
            v53 = v6 == 32;
            goto LABEL_4;
          }
        }
        v60 = 0;
        v61 = v4;
        v62 = (__int64 **)v4;
        v29 = v5[2];
        if ( v29 && !*(_QWORD *)(v29 + 8) && *(_BYTE *)v5 == 57 )
        {
          v43 = (_BYTE *)*(v5 - 8);
          if ( *v43 == 44 )
          {
            v56 = *(v5 - 8);
            v49 = sub_10081F0(&v60, *((_QWORD *)v43 - 8));
            v44 = (__int64 **)*(v5 - 4);
            if ( v49 && *(__int64 **)(v56 - 32) == v61 && v44 == v62 )
              goto LABEL_81;
          }
          else
          {
            v44 = (__int64 **)*(v5 - 4);
          }
          if ( *(_BYTE *)v44 == 44
            && (unsigned __int8)sub_10081F0(&v60, (__int64)*(v44 - 8))
            && *(v44 - 4) == v61
            && (__int64 **)*(v5 - 8) == v62 )
          {
LABEL_81:
            v57 = v4;
            v5 = v4;
            goto LABEL_33;
          }
        }
        v5 = v57;
LABEL_33:
        v53 = v6 == 32;
        goto LABEL_4;
      }
    }
    else
    {
      v33 = (__int64 *)*(v4 - 4);
    }
    if ( *(_BYTE *)v33 != 42 || !(unsigned __int8)sub_1111CE0(&v60, (__int64)v33) || (__int64 *)*(v4 - 8) != *v62 )
      goto LABEL_29;
    goto LABEL_52;
  }
  if ( !sub_B532A0(v6) )
  {
LABEL_3:
    v5 = v57;
    goto LABEL_4;
  }
  if ( (unsigned int)(v6 - 35) > 1 )
  {
    v53 = v6 == 37;
    if ( v6 != 34 && v6 != 37 )
      goto LABEL_3;
    v60 = v4;
    v61 = 0;
    v62 = (__int64 **)v4;
    v24 = v5[2];
    if ( !v24 || *(_QWORD *)(v24 + 8) || *(_BYTE *)v5 != 59 )
      goto LABEL_3;
    v25 = (_BYTE *)*(v5 - 8);
    if ( *v25 == 42 && v4 == *((__int64 **)v25 - 8) )
    {
      v50 = (unsigned __int8)sub_995B10(&v61, *((_QWORD *)v25 - 4)) == 0;
      v26 = (__int64 **)*(v5 - 4);
      if ( !v50 && v26 == v62 )
      {
LABEL_27:
        v57 = v4;
        v5 = v4;
        goto LABEL_4;
      }
    }
    else
    {
      v26 = (__int64 **)*(v5 - 4);
    }
    if ( *(_BYTE *)v26 != 42
      || *(v26 - 8) != v60
      || !(unsigned __int8)sub_995B10(&v61, (__int64)*(v26 - 4))
      || (__int64 **)*(v5 - 8) != v62 )
    {
      goto LABEL_3;
    }
    goto LABEL_27;
  }
  v60 = v5;
  v61 = 0;
  v62 = (__int64 **)v5;
  v37 = v4[2];
  if ( !v37 || *(_QWORD *)(v37 + 8) || *(_BYTE *)v4 != 59 )
    goto LABEL_3;
  v38 = (__int64 **)*(v4 - 8);
  if ( *(_BYTE *)v38 != 42 || v5 != *(v38 - 8) || !(unsigned __int8)sub_995B10(&v61, (__int64)*(v38 - 4)) )
  {
    v39 = (__int64 **)*(v4 - 4);
LABEL_62:
    if ( *(_BYTE *)v39 != 42
      || *(v39 - 8) != v60
      || !(unsigned __int8)sub_995B10(&v61, (__int64)*(v39 - 4))
      || (__int64 **)*(v4 - 8) != v62 )
    {
      goto LABEL_3;
    }
    goto LABEL_66;
  }
  v39 = (__int64 **)*(v4 - 4);
  if ( v39 != v62 )
    goto LABEL_62;
LABEL_66:
  v57 = v5;
  v53 = v6 == 35;
LABEL_4:
  if ( !v5 )
    return v5;
  v7 = v5[1];
  v63 = 257;
  HIDWORD(v59) = 0;
  v8 = sub_B33BC0(a2, 0x42u, (__int64)v5, (unsigned int)v59, (__int64)&v60);
  if ( v53 )
  {
    v9 = sub_AD64C0(v7, 2, 0);
    v63 = 257;
    v5 = sub_BD2C40(72, unk_3F10FD0);
    if ( v5 )
    {
      v10 = *(_QWORD *)(v8 + 8);
      v11 = *(unsigned __int8 *)(v10 + 8);
      if ( (unsigned int)(v11 - 17) > 1 )
      {
        v15 = sub_BCB2A0(*(_QWORD **)v10);
      }
      else
      {
        v12 = *(_DWORD *)(v10 + 32);
        v13 = *(_QWORD **)v10;
        BYTE4(v58) = (_BYTE)v11 == 18;
        LODWORD(v58) = v12;
        v14 = (__int64 *)sub_BCB2A0(v13);
        v15 = sub_BCE1B0(v14, v58);
      }
      sub_B523C0((__int64)v5, v15, 53, 36, v8, v9, (__int64)&v60, 0, 0, 0);
      return v5;
    }
    return 0;
  }
  v17 = sub_AD64C0(v7, 1, 0);
  v63 = 257;
  v5 = sub_BD2C40(72, unk_3F10FD0);
  if ( !v5 )
    return 0;
  v18 = *(_QWORD *)(v8 + 8);
  v19 = *(unsigned __int8 *)(v18 + 8);
  if ( (unsigned int)(v19 - 17) > 1 )
  {
    v23 = sub_BCB2A0(*(_QWORD **)v18);
  }
  else
  {
    v20 = *(_DWORD *)(v18 + 32);
    v21 = *(_QWORD **)v18;
    BYTE4(v59) = (_BYTE)v19 == 18;
    LODWORD(v59) = v20;
    v22 = (__int64 *)sub_BCB2A0(v21);
    v23 = sub_BCE1B0(v22, v59);
  }
  sub_B523C0((__int64)v5, v23, 53, 34, v8, v17, (__int64)&v60, 0, 0, 0);
  return v5;
}
