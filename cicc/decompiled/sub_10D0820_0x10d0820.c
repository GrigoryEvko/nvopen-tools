// Function: sub_10D0820
// Address: 0x10d0820
//
__int64 __fastcall sub_10D0820(__int64 **a1, int a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v6; // r12
  _BYTE *v7; // rbx
  char v8; // al
  _BYTE *v9; // r15
  __int64 v10; // rcx
  unsigned int v11; // r14d
  int v12; // eax
  __int64 v13; // rdx
  _BYTE *v14; // r12
  unsigned int v15; // ebx
  bool v16; // al
  __int64 *v17; // rax
  _BYTE *v18; // r15
  __int64 v19; // rcx
  unsigned int v20; // ebx
  __int64 v21; // rdi
  int v22; // eax
  bool v23; // al
  int v24; // eax
  __int64 v25; // rcx
  unsigned int v26; // r15d
  __int64 v27; // rdi
  int v28; // eax
  bool v29; // al
  __int64 *v30; // rax
  __int64 v31; // r15
  __int64 v32; // rdx
  _BYTE *v33; // rax
  __int64 v34; // rbx
  __int64 v35; // rdx
  _BYTE *v36; // rax
  unsigned int v37; // ebx
  __int64 v38; // rbx
  _BYTE *v39; // rax
  __int64 v40; // r14
  _BYTE *v41; // rax
  unsigned int v42; // r14d
  int v43; // eax
  bool v44; // al
  int v45; // eax
  __int64 v46; // rsi
  bool v47; // r15
  __int64 v48; // rax
  unsigned int v49; // r15d
  int v50; // eax
  bool v51; // bl
  unsigned int v52; // r15d
  __int64 v53; // rax
  unsigned int v54; // ebx
  int v55; // eax
  __int64 v56; // rsi
  bool v57; // bl
  __int64 v58; // rax
  unsigned int v59; // ebx
  int v60; // eax
  bool v61; // r14
  __int64 v62; // rsi
  __int64 v63; // rax
  unsigned int v64; // r14d
  int v65; // eax
  int v66; // [rsp+Ch] [rbp-44h]
  int v67; // [rsp+Ch] [rbp-44h]
  int v68; // [rsp+Ch] [rbp-44h]
  __int64 v69; // [rsp+18h] [rbp-38h]
  __int64 v70; // [rsp+18h] [rbp-38h]
  __int64 v71; // [rsp+18h] [rbp-38h]
  __int64 v72; // [rsp+18h] [rbp-38h]
  __int64 v73; // [rsp+18h] [rbp-38h]
  int v74; // [rsp+18h] [rbp-38h]
  __int64 v75; // [rsp+18h] [rbp-38h]
  __int64 v76; // [rsp+18h] [rbp-38h]

  if ( a2 + 29 != *(unsigned __int8 *)a3 )
    goto LABEL_2;
  v3 = *(_QWORD *)(a3 - 64);
  v6 = a3;
  if ( *(_BYTE *)v3 != 42 )
    goto LABEL_5;
  v18 = *(_BYTE **)(v3 - 64);
  if ( *v18 != 54 )
    goto LABEL_5;
  v19 = *((_QWORD *)v18 - 8);
  if ( *(_BYTE *)v19 == 17 )
  {
    v20 = *(_DWORD *)(v19 + 32);
    if ( v20 > 0x40 )
    {
      v70 = *((_QWORD *)v18 - 8);
      v21 = v19 + 24;
LABEL_27:
      v22 = sub_C444A0(v21);
      v19 = v70;
      v23 = v20 - 1 == v22;
      goto LABEL_28;
    }
    v23 = *(_QWORD *)(v19 + 24) == 1;
  }
  else
  {
    v38 = *(_QWORD *)(v19 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v38 + 8) - 17 > 1 || *(_BYTE *)v19 > 0x15u )
      goto LABEL_5;
    v70 = *((_QWORD *)v18 - 8);
    v39 = sub_AD7630(v70, 0, a3);
    v19 = v70;
    if ( !v39 || *v39 != 17 )
    {
      if ( *(_BYTE *)(v38 + 8) == 17 )
      {
        v55 = *(_DWORD *)(v38 + 32);
        v56 = 0;
        v57 = 0;
        v67 = v55;
        if ( v55 )
        {
          while ( 1 )
          {
            v75 = v19;
            v58 = sub_AD69F0((unsigned __int8 *)v19, v56);
            v19 = v75;
            if ( !v58 )
              break;
            if ( *(_BYTE *)v58 != 13 )
            {
              if ( *(_BYTE *)v58 != 17 )
                break;
              v59 = *(_DWORD *)(v58 + 32);
              if ( v59 <= 0x40 )
              {
                v57 = *(_QWORD *)(v58 + 24) == 1;
              }
              else
              {
                v60 = sub_C444A0(v58 + 24);
                v19 = v75;
                v57 = v59 - 1 == v60;
              }
              if ( !v57 )
                break;
            }
            v56 = (unsigned int)(v56 + 1);
            if ( v67 == (_DWORD)v56 )
            {
              if ( v57 )
                goto LABEL_29;
              goto LABEL_5;
            }
          }
        }
      }
      goto LABEL_5;
    }
    v20 = *((_DWORD *)v39 + 8);
    if ( v20 > 0x40 )
    {
      v21 = (__int64)(v39 + 24);
      goto LABEL_27;
    }
    v23 = *((_QWORD *)v39 + 3) == 1;
  }
LABEL_28:
  if ( !v23 )
  {
LABEL_5:
    v7 = *(_BYTE **)(v6 - 32);
    v8 = *v7;
    goto LABEL_6;
  }
LABEL_29:
  if ( *a1 )
    **a1 = v19;
  a3 = *((_QWORD *)v18 - 4);
  if ( !a3 )
    goto LABEL_5;
  *a1[1] = a3;
  v24 = sub_995B10(a1 + 2, *(_QWORD *)(v3 - 32));
  v7 = *(_BYTE **)(v6 - 32);
  LODWORD(v3) = v24;
  v8 = *v7;
  if ( !(_BYTE)v3 || v8 != 54 )
  {
LABEL_6:
    if ( v8 != 42 )
      goto LABEL_2;
    v9 = (_BYTE *)*((_QWORD *)v7 - 8);
    if ( *v9 != 54 )
      goto LABEL_2;
    v10 = *((_QWORD *)v9 - 8);
    if ( *(_BYTE *)v10 == 17 )
    {
      v11 = *(_DWORD *)(v10 + 32);
      if ( v11 <= 0x40 )
      {
        if ( *(_QWORD *)(v10 + 24) != 1 )
          goto LABEL_2;
      }
      else
      {
        v69 = *((_QWORD *)v9 - 8);
        v12 = sub_C444A0(v10 + 24);
        v10 = v69;
        if ( v12 != v11 - 1 )
          goto LABEL_2;
      }
    }
    else
    {
      v40 = *(_QWORD *)(v10 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v40 + 8) - 17 > 1 || *(_BYTE *)v10 > 0x15u )
        goto LABEL_2;
      v72 = *((_QWORD *)v9 - 8);
      v41 = sub_AD7630(v72, 0, a3);
      v10 = v72;
      if ( !v41 || *v41 != 17 )
      {
        if ( *(_BYTE *)(v40 + 8) == 17 )
        {
          v68 = *(_DWORD *)(v40 + 32);
          if ( v68 )
          {
            v61 = 0;
            v62 = 0;
            while ( 1 )
            {
              v76 = v10;
              v63 = sub_AD69F0((unsigned __int8 *)v10, v62);
              if ( !v63 )
                break;
              v10 = v76;
              if ( *(_BYTE *)v63 != 13 )
              {
                if ( *(_BYTE *)v63 != 17 )
                  break;
                v64 = *(_DWORD *)(v63 + 32);
                if ( v64 <= 0x40 )
                {
                  v61 = *(_QWORD *)(v63 + 24) == 1;
                }
                else
                {
                  v65 = sub_C444A0(v63 + 24);
                  v10 = v76;
                  v61 = v64 - 1 == v65;
                }
                if ( !v61 )
                  break;
              }
              v62 = (unsigned int)(v62 + 1);
              if ( v68 == (_DWORD)v62 )
              {
                if ( v61 )
                  goto LABEL_11;
                goto LABEL_2;
              }
            }
          }
        }
        goto LABEL_2;
      }
      v42 = *((_DWORD *)v41 + 8);
      if ( v42 <= 0x40 )
      {
        v44 = *((_QWORD *)v41 + 3) == 1;
      }
      else
      {
        v43 = sub_C444A0((__int64)(v41 + 24));
        v10 = v72;
        v44 = v42 - 1 == v43;
      }
      if ( !v44 )
        goto LABEL_2;
    }
LABEL_11:
    if ( *a1 )
      **a1 = v10;
    v13 = *((_QWORD *)v9 - 4);
    if ( !v13 )
      goto LABEL_2;
    *a1[1] = v13;
    if ( !(unsigned __int8)sub_995B10(a1 + 2, *((_QWORD *)v7 - 4)) )
      goto LABEL_2;
    v14 = *(_BYTE **)(v6 - 64);
    if ( *v14 != 54 )
      goto LABEL_2;
    v3 = *((_QWORD *)v14 - 8);
    if ( *(_BYTE *)v3 == 17 )
    {
      v15 = *(_DWORD *)(v3 + 32);
      if ( v15 <= 0x40 )
        v16 = *(_QWORD *)(v3 + 24) == 1;
      else
        v16 = v15 - 1 == (unsigned int)sub_C444A0(v3 + 24);
    }
    else
    {
      v34 = *(_QWORD *)(v3 + 8);
      v35 = (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17;
      if ( (unsigned int)v35 > 1 || *(_BYTE *)v3 > 0x15u )
        goto LABEL_2;
      v36 = sub_AD7630(*((_QWORD *)v14 - 8), 0, v35);
      if ( !v36 || *v36 != 17 )
      {
        if ( *(_BYTE *)(v34 + 8) == 17 )
        {
          v74 = *(_DWORD *)(v34 + 32);
          if ( v74 )
          {
            v51 = 0;
            v52 = 0;
            while ( 1 )
            {
              v53 = sub_AD69F0((unsigned __int8 *)v3, v52);
              if ( !v53 )
                break;
              if ( *(_BYTE *)v53 != 13 )
              {
                if ( *(_BYTE *)v53 != 17 )
                  break;
                v54 = *(_DWORD *)(v53 + 32);
                v51 = v54 <= 0x40 ? *(_QWORD *)(v53 + 24) == 1 : v54 - 1 == (unsigned int)sub_C444A0(v53 + 24);
                if ( !v51 )
                  break;
              }
              if ( v74 == ++v52 )
              {
                if ( v51 )
                  goto LABEL_20;
                goto LABEL_2;
              }
            }
          }
        }
        goto LABEL_2;
      }
      v37 = *((_DWORD *)v36 + 8);
      if ( v37 <= 0x40 )
        v16 = *((_QWORD *)v36 + 3) == 1;
      else
        v16 = v37 - 1 == (unsigned int)sub_C444A0((__int64)(v36 + 24));
    }
    if ( v16 )
    {
LABEL_20:
      v17 = a1[3];
      if ( v17 )
        *v17 = v3;
      LOBYTE(v3) = *a1[4] == *((_QWORD *)v14 - 4);
      return (unsigned int)v3;
    }
LABEL_2:
    LODWORD(v3) = 0;
    return (unsigned int)v3;
  }
  v25 = *((_QWORD *)v7 - 8);
  if ( *(_BYTE *)v25 == 17 )
  {
    v26 = *(_DWORD *)(v25 + 32);
    if ( v26 > 0x40 )
    {
      v71 = *((_QWORD *)v7 - 8);
      v27 = v25 + 24;
LABEL_37:
      v28 = sub_C444A0(v27);
      v25 = v71;
      v29 = v26 - 1 == v28;
      goto LABEL_38;
    }
    v29 = *(_QWORD *)(v25 + 24) == 1;
  }
  else
  {
    v31 = *(_QWORD *)(v25 + 8);
    v32 = (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17;
    if ( (unsigned int)v32 > 1 || *(_BYTE *)v25 > 0x15u )
      goto LABEL_2;
    v71 = *((_QWORD *)v7 - 8);
    v33 = sub_AD7630(v71, 0, v32);
    v25 = v71;
    if ( !v33 || *v33 != 17 )
    {
      if ( *(_BYTE *)(v31 + 8) == 17 )
      {
        v45 = *(_DWORD *)(v31 + 32);
        v46 = 0;
        v47 = 0;
        v66 = v45;
        if ( v45 )
        {
          while ( 1 )
          {
            v73 = v25;
            v48 = sub_AD69F0((unsigned __int8 *)v25, v46);
            v25 = v73;
            if ( !v48 )
              break;
            if ( *(_BYTE *)v48 != 13 )
            {
              if ( *(_BYTE *)v48 != 17 )
                break;
              v49 = *(_DWORD *)(v48 + 32);
              if ( v49 <= 0x40 )
              {
                v47 = *(_QWORD *)(v48 + 24) == 1;
              }
              else
              {
                v50 = sub_C444A0(v48 + 24);
                a3 = v49 - 1;
                v25 = v73;
                v47 = (_DWORD)a3 == v50;
              }
              if ( !v47 )
                break;
            }
            v46 = (unsigned int)(v46 + 1);
            if ( v66 == (_DWORD)v46 )
            {
              if ( v47 )
                goto LABEL_39;
              goto LABEL_5;
            }
          }
        }
      }
      goto LABEL_5;
    }
    v26 = *((_DWORD *)v33 + 8);
    if ( v26 > 0x40 )
    {
      v27 = (__int64)(v33 + 24);
      goto LABEL_37;
    }
    v29 = *((_QWORD *)v33 + 3) == 1;
  }
LABEL_38:
  if ( !v29 )
    goto LABEL_5;
LABEL_39:
  v30 = a1[3];
  if ( v30 )
    *v30 = v25;
  if ( *((_QWORD *)v7 - 4) != *a1[4] )
    goto LABEL_5;
  return (unsigned int)v3;
}
