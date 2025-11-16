// Function: sub_AA93C0
// Address: 0xaa93c0
//
__int64 __fastcall sub_AA93C0(unsigned int a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int8 *v5; // rbx
  int v6; // eax
  unsigned __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int8 v11; // al
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // r15
  int v16; // eax
  unsigned int v17; // r12d
  __int64 v18; // r15
  __int64 v19; // rax
  unsigned int v20; // r12d
  __int64 v21; // rax
  unsigned __int8 *v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // r15
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rax
  _BYTE *v31; // rbx
  _BOOL8 v32; // rdx
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned int v35; // eax
  __int64 v36; // r12
  int v37; // eax
  unsigned int v38; // r14d
  __int64 v39; // rdi
  __int64 v40; // r14
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // rax
  unsigned __int8 *v45; // rdx
  unsigned int v46; // eax
  unsigned __int8 v47; // cl
  int v48; // edx
  int v49; // r14d
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // rax
  unsigned __int8 *v54; // rsi
  __int64 v55; // rax
  unsigned int v56; // r13d
  __int64 v57; // rax
  __int64 v58; // r13
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  unsigned int v61; // eax
  __int64 v62; // rax
  __int64 v63; // r15
  __int64 v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // [rsp+10h] [rbp-E0h]
  __int64 v70; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v71; // [rsp+18h] [rbp-D8h]
  int v72; // [rsp+18h] [rbp-D8h]
  __int64 v73; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v74; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v75; // [rsp+28h] [rbp-C8h]
  _BYTE *v76; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v77; // [rsp+38h] [rbp-B8h]
  _BYTE v78[176]; // [rsp+40h] [rbp-B0h] BYREF

  v5 = (unsigned __int8 *)a2;
  while ( 1 )
  {
    v6 = *v5;
    if ( (_BYTE)v6 == 13 )
      return sub_ACADE0(a3);
    if ( (unsigned int)(v6 - 12) <= 1 )
    {
      if ( ((a1 - 39) & 0xFFFFFFFA) != 0 )
        return sub_ACA8A0(a3);
      else
        return sub_AD6530(a3);
    }
    v8 = (unsigned __int64)v5;
    if ( (unsigned __int8)sub_AC30F0(v5) && *(_BYTE *)(a3 + 8) != 10 && a1 != 50 )
      return sub_AD6530(a3);
    v11 = *v5;
    if ( *v5 != 5 )
      goto LABEL_12;
    v8 = (unsigned __int64)v5;
    if ( !(unsigned __int8)sub_AC35E0(v5) )
      break;
    v17 = *((unsigned __int16 *)v5 + 1);
    v18 = *((_QWORD *)v5 + 1);
    v70 = *(_QWORD *)(*(_QWORD *)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)] + 8LL);
    v19 = sub_BCB2E0(*(_QWORD *)a3);
    v8 = v17;
    a2 = a1;
    v20 = sub_B50810(v17, a1, v70, v18, a3, 0, v19, 0);
    if ( !v20 )
      break;
    v5 = *(unsigned __int8 **)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
    if ( (unsigned __int8)sub_AC4810(v20) )
      return sub_ADAB70(v20, v5, a3, 0);
    a1 = v20;
  }
  v11 = *v5;
LABEL_12:
  if ( v11 != 16 && v11 != 11
    || (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 > 1
    || (v9 = *(unsigned int *)(*((_QWORD *)v5 + 1) + 32LL), *(_DWORD *)(a3 + 32) != (_DWORD)v9) )
  {
    switch ( a1 )
    {
      case '&':
        v15 = 0;
        if ( v11 != 17 )
          return v15;
        sub_BCB060(a3);
        sub_C44740(&v76, v5 + 24);
        goto LABEL_52;
      case '\'':
        v15 = 0;
        if ( v11 != 17 )
          return v15;
        v35 = sub_BCB060(a3);
        sub_C449B0(&v76, v5 + 24, v35);
        goto LABEL_52;
      case '(':
        v15 = 0;
        if ( v11 != 17 )
          return v15;
        v46 = sub_BCB060(a3);
        sub_C44830(&v76, v5 + 24, v46);
        goto LABEL_52;
      case ')':
      case '*':
        v15 = 0;
        if ( v11 != 18 )
          return v15;
        LODWORD(v77) = sub_BCB060(a3);
        if ( (unsigned int)v77 > 0x40 )
          sub_C43690(&v76, 0, 0);
        else
          v76 = 0;
        BYTE4(v77) = a1 == 41;
        if ( (unsigned int)sub_C41980(v5 + 24, &v76, 0, &v74, v33, v34) != 1 )
          goto LABEL_52;
        v15 = sub_ACADE0(a3);
        goto LABEL_53;
      case '+':
      case ',':
        v15 = 0;
        if ( v11 != 17 )
          return v15;
        v71 = v5 + 24;
        v75 = sub_BCB060(a3);
        if ( v75 > 0x40 )
        {
          a2 = 0;
          sub_C43690(&v74, 0, 0);
        }
        else
        {
          v74 = 0;
        }
        v25 = a3;
        if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
          v25 = **(_QWORD **)(a3 + 16);
        v26 = sub_BCAC60(v25);
        v30 = sub_C33340(v25, a2, v27, v28, v29);
        v31 = (_BYTE *)v30;
        if ( v26 == v30 )
          sub_C3C640(&v76, v30, &v74);
        else
          sub_C3B160(&v76, v26, &v74);
        if ( v75 > 0x40 && v74 )
          j_j___libc_free_0_0(v74);
        v32 = a1 == 44;
        if ( v76 == v31 )
          sub_C400C0(&v76, v71, v32, 1);
        else
          sub_C36910(&v76, v71, v32, 1);
        goto LABEL_35;
      case '-':
      case '.':
        v15 = 0;
        if ( v11 != 18 )
          return v15;
        v21 = sub_C33340(v8, a2, a1 - 38, v9, v10);
        v22 = v5 + 24;
        if ( *((_QWORD *)v5 + 3) == v21 )
          sub_C3C790(&v76, v22);
        else
          sub_C33EB0(&v76, v22);
        v23 = a3;
        if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
          v23 = **(_QWORD **)(a3 + 16);
        v24 = sub_BCAC60(v23);
        sub_C41640(&v76, v24, 1, &v74);
        goto LABEL_35;
      case '/':
      case '0':
      case '2':
        return 0;
      case '1':
        v74 = v5;
        v36 = *((_QWORD *)v5 + 1);
        v15 = (__int64)v5;
        if ( a3 == v36 )
          return v15;
        if ( (unsigned __int8)sub_AD7930(v5) )
          return sub_AD62B0(a3);
        if ( *v74 == 17 )
        {
          v37 = *(unsigned __int8 *)(a3 + 8);
          v38 = v37 - 17;
          if ( (unsigned int)(v37 - 17) > 1 )
          {
            if ( v37 != 18 )
              goto LABEL_64;
            goto LABEL_63;
          }
          if ( (unsigned int)*(unsigned __int8 *)(v36 + 8) - 17 <= 1 )
          {
LABEL_63:
            LOBYTE(v37) = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
LABEL_64:
            if ( (unsigned __int8)v37 <= 3u || (_BYTE)v37 == 5 || (v15 = 0, (v37 & 0xFD) == 4) )
            {
              if ( *(_BYTE *)(a3 + 8) == 6 )
              {
                return 0;
              }
              else
              {
                v15 = 0;
                v72 = sub_BCB060(a3);
                if ( v72 == (unsigned int)sub_BCB060(v36) )
                {
                  v39 = a3;
                  if ( v38 <= 1 )
                    v39 = **(_QWORD **)(a3 + 16);
                  v40 = sub_BCAC60(v39);
                  v44 = sub_C33340(v39, a2, v41, v42, v43);
                  v45 = v74 + 24;
                  if ( v40 == v44 )
                    sub_C3C640(&v76, v40, v45);
                  else
                    sub_C3B160(&v76, v40, v45);
LABEL_35:
                  v15 = sub_AD8F10(a3, &v76);
                  sub_91D830(&v76);
                }
              }
            }
            return v15;
          }
LABEL_111:
          v65 = sub_AD3730(&v74, 1);
          return sub_AD4C90(v65, a3, 0, v66, v67, v68);
        }
        v15 = 0;
        if ( *v74 != 18 )
          return v15;
        v47 = *(_BYTE *)(v36 + 8);
        v48 = *(unsigned __int8 *)(a3 + 8);
        if ( (unsigned int)(v48 - 17) > 1 )
        {
          if ( v47 == 6 )
            return v15;
        }
        else
        {
          if ( (unsigned int)v47 - 17 > 1 )
            goto LABEL_111;
          if ( v47 == 6 )
            return v15;
          LOBYTE(v48) = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
        }
        v15 = 0;
        if ( (_BYTE)v48 == 12 )
        {
          v49 = sub_BCB060(a3);
          if ( v49 == (unsigned int)sub_BCB060(v36) )
          {
            v53 = sub_C33340(v36, a2, v50, v51, v52);
            v54 = v74 + 24;
            if ( *((_QWORD *)v74 + 3) == v53 )
              sub_C3E660(&v76, v54);
            else
              sub_C3A850(&v76, v54);
LABEL_52:
            v15 = sub_AD8D80(a3, &v76);
LABEL_53:
            if ( (unsigned int)v77 > 0x40 )
            {
              if ( v76 )
                j_j___libc_free_0_0(v76);
            }
          }
        }
        return v15;
      default:
        BUG();
    }
  }
  v12 = *(_QWORD *)(a3 + 24);
  v14 = sub_AD7630(v5, 0);
  if ( v14 )
  {
    if ( (unsigned __int8)sub_AC4810(a1) )
      v15 = sub_ADAB70(a1, v14, v12, 0);
    else
      v15 = sub_AA93C0(a1, v14, v12);
    if ( v15 )
    {
      v16 = *(_DWORD *)(a3 + 32);
      BYTE4(v76) = *(_BYTE *)(a3 + 8) == 18;
      LODWORD(v76) = v16;
      return sub_AD5E10((size_t)v76);
    }
  }
  else
  {
    v76 = v78;
    v77 = 0x1000000000LL;
    v55 = sub_BD5C60(v5, 0, v13);
    v73 = sub_BCCE00(v55, 32);
    v56 = *(_DWORD *)(*((_QWORD *)v5 + 1) + 32LL);
    if ( v56 )
    {
      v57 = v56;
      v58 = 0;
      v69 = v57;
      while ( 1 )
      {
        v62 = sub_AD64C0(v73, v58, 0);
        v63 = sub_AD5840(v5, v62, 0);
        v64 = v63;
        v15 = (unsigned __int8)sub_AC4810(a1) ? sub_ADAB70(a1, v63, v12, 0) : sub_AA93C0(a1, v63, v12);
        if ( !v15 )
          break;
        v59 = (unsigned int)v77;
        v60 = (unsigned int)v77 + 1LL;
        if ( v60 > HIDWORD(v77) )
        {
          sub_C8D5F0(&v76, v78, v60, 8);
          v59 = (unsigned int)v77;
        }
        ++v58;
        *(_QWORD *)&v76[8 * v59] = v15;
        v61 = v77 + 1;
        LODWORD(v77) = v77 + 1;
        if ( v69 == v58 )
          goto LABEL_106;
      }
    }
    else
    {
      v61 = v77;
LABEL_106:
      v64 = v61;
      v15 = sub_AD3730(v76, v61);
    }
    if ( v76 != v78 )
      _libc_free(v76, v64);
  }
  return v15;
}
