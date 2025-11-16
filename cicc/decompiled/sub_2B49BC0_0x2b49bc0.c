// Function: sub_2B49BC0
// Address: 0x2b49bc0
//
__int64 __fastcall sub_2B49BC0(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 v3; // al
  char v4; // di
  __int64 v5; // rsi
  int v6; // edx
  __int64 v7; // rax
  unsigned int v8; // ecx
  __int64 v9; // rax
  unsigned __int8 *v10; // r8
  __int64 v11; // rdx
  unsigned int v12; // r14d
  __int64 v14; // r13
  unsigned __int8 *v15; // r15
  unsigned int v16; // r12d
  unsigned int v17; // eax
  unsigned __int8 *v18; // r14
  unsigned __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rdx
  unsigned int v25; // eax
  char v26; // cl
  __int64 *v27; // rdx
  __int64 *v28; // r13
  __int64 v29; // rax
  __int64 *v30; // r12
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 *v36; // r10
  __int64 v37; // r9
  unsigned int v38; // eax
  __int64 *v39; // r12
  __int64 v40; // r13
  __int64 *v41; // r14
  __int64 v42; // r15
  __int64 *v43; // rax
  __int64 v44; // r11
  int v45; // eax
  __int64 v46; // r15
  char v47; // di
  int v48; // edi
  __int64 v49; // r8
  int v50; // esi
  unsigned int v51; // ecx
  __int64 *v52; // rdx
  __int64 v53; // r9
  __int64 *v54; // rdx
  unsigned int v55; // esi
  unsigned int v56; // edx
  int v57; // ecx
  unsigned int v58; // r8d
  int v59; // r11d
  __int64 *v60; // r10
  __int64 v61; // r12
  __int64 v62; // rdx
  int v63; // eax
  __int64 v64; // rdi
  __int64 v65; // rax
  unsigned __int64 v66; // rax
  int *v67; // rdi
  int v68; // r15d
  int v69; // r9d
  __int64 v70; // [rsp+0h] [rbp-160h]
  __int64 v71; // [rsp+0h] [rbp-160h]
  __int64 v72; // [rsp+10h] [rbp-150h]
  unsigned int v73; // [rsp+18h] [rbp-148h]
  unsigned int v74; // [rsp+1Ch] [rbp-144h]
  __int64 v76; // [rsp+28h] [rbp-138h]
  _BYTE *v77; // [rsp+28h] [rbp-138h]
  __int64 v78; // [rsp+38h] [rbp-128h] BYREF
  __int64 *v79; // [rsp+40h] [rbp-120h] BYREF
  __int64 v80; // [rsp+48h] [rbp-118h]
  int *v81; // [rsp+50h] [rbp-110h] BYREF
  __int64 v82; // [rsp+58h] [rbp-108h]
  int v83; // [rsp+60h] [rbp-100h] BYREF
  __int64 v84; // [rsp+68h] [rbp-F8h]
  unsigned __int8 *v85; // [rsp+70h] [rbp-F0h]
  __int64 v86; // [rsp+90h] [rbp-D0h] BYREF
  __int64 *v87; // [rsp+98h] [rbp-C8h]
  __int64 v88; // [rsp+A0h] [rbp-C0h]
  int v89; // [rsp+A8h] [rbp-B8h]
  char v90; // [rsp+ACh] [rbp-B4h]
  unsigned __int8 *v91; // [rsp+B0h] [rbp-B0h] BYREF

  while ( 1 )
  {
    v3 = *a2;
    if ( *a2 <= 0x1Cu )
      break;
    if ( v3 == 62 )
    {
      v86 = sub_9208B0(*(_QWORD *)(a1 + 3344), *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL));
      v87 = (__int64 *)v31;
      return (unsigned int)sub_CA1930(&v86);
    }
    if ( v3 != 91 )
      break;
    a2 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
    if ( !a2 )
      BUG();
  }
  v4 = *(_BYTE *)(a1 + 696) & 1;
  if ( (*(_BYTE *)(a1 + 696) & 1) != 0 )
  {
    v5 = a1 + 704;
    v6 = 3;
  }
  else
  {
    v7 = *(unsigned int *)(a1 + 712);
    v5 = *(_QWORD *)(a1 + 704);
    if ( !(_DWORD)v7 )
      goto LABEL_41;
    v6 = v7 - 1;
  }
  v8 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = v5 + 16LL * v8;
  v10 = *(unsigned __int8 **)v9;
  if ( a2 == *(unsigned __int8 **)v9 )
    goto LABEL_9;
  v63 = 1;
  while ( v10 != (unsigned __int8 *)-4096LL )
  {
    v69 = v63 + 1;
    v8 = v6 & (v63 + v8);
    v9 = v5 + 16LL * v8;
    v10 = *(unsigned __int8 **)v9;
    if ( *(unsigned __int8 **)v9 == a2 )
      goto LABEL_9;
    v63 = v69;
  }
  if ( v4 )
  {
    v32 = 64;
    goto LABEL_42;
  }
  v7 = *(unsigned int *)(a1 + 712);
LABEL_41:
  v32 = 16 * v7;
LABEL_42:
  v9 = v5 + v32;
LABEL_9:
  v11 = 64;
  if ( !v4 )
    v11 = 16LL * *(unsigned int *)(a1 + 712);
  if ( v9 != v5 + v11 )
    return *(unsigned int *)(v9 + 8);
  v86 = 0;
  v81 = &v83;
  v82 = 0x200000000LL;
  v87 = (__int64 *)&v91;
  v88 = 16;
  v89 = 0;
  v90 = 1;
  if ( *a2 <= 0x1Cu )
  {
    sub_BCB2A0(*(_QWORD **)(a1 + 3440));
    v15 = a2;
LABEL_102:
    v79 = (__int64 *)sub_9208B0(*(_QWORD *)(a1 + 3344), *((_QWORD *)v15 + 1));
    v80 = v62;
    v12 = sub_CA1930(&v79);
    goto LABEL_30;
  }
  v14 = *((_QWORD *)a2 + 5);
  v85 = a2;
  v15 = 0;
  v16 = 0;
  v91 = a2;
  v17 = 1;
  v74 = 0;
  v18 = a2;
  v83 = 0;
  v84 = v14;
  HIDWORD(v88) = 1;
  v86 = 1;
  while ( 1 )
  {
    LODWORD(v82) = --v17;
    v76 = *((_QWORD *)v18 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v76 + 8) - 17 <= 1 )
    {
      if ( !v17 )
        goto LABEL_29;
      goto LABEL_17;
    }
    if ( v76 != sub_BCB2A0(*(_QWORD **)(a1 + 3440)) && !v15 )
      v15 = v18;
    if ( v16 <= (unsigned int)qword_500FC48 )
      break;
LABEL_28:
    v17 = v82;
    if ( !(_DWORD)v82 )
      goto LABEL_29;
LABEL_17:
    v19 = (unsigned __int64)&v81[6 * v17 - 6];
    v16 = *(_DWORD *)v19;
    v14 = *(_QWORD *)(v19 + 8);
    v18 = *(unsigned __int8 **)(v19 + 16);
  }
  v22 = *v18;
  v23 = (unsigned __int8)(v22 - 61) & 0xDF;
  if ( (((_BYTE)v22 - 61) & 0xDF) == 0 || (_BYTE)v22 == 90 )
  {
    v79 = (__int64 *)sub_9208B0(*(_QWORD *)(a1 + 3344), v76);
    v80 = v24;
    v25 = sub_CA1930(&v79);
    if ( v74 >= v25 )
      v25 = v74;
    v74 = v25;
    goto LABEL_28;
  }
  if ( (_BYTE)v22 == 84
    || (v33 = (unsigned int)(v22 - 41), (unsigned __int8)v33 <= 0x2Du)
    && (v34 = 0x267FFC47FFFFLL, _bittest64(&v34, v33)) )
  {
    v35 = 4LL * (*((_DWORD *)v18 + 1) & 0x7FFFFFF);
    if ( (v18[7] & 0x40) != 0 )
    {
      v36 = (__int64 *)*((_QWORD *)v18 - 1);
      v37 = (__int64)&v36[v35];
    }
    else
    {
      v37 = (__int64)v18;
      v36 = (__int64 *)&v18[-(v35 * 8)];
    }
    if ( (__int64 *)v37 == v36 )
      goto LABEL_28;
    v38 = v16 + 1;
    v72 = v14;
    v39 = v36;
    v40 = (__int64)v15;
    v73 = v38;
    v77 = v18;
    v41 = (__int64 *)v37;
    while ( 2 )
    {
      v42 = *v39;
      if ( *(_BYTE *)*v39 > 0x1Cu )
      {
        if ( !v90 )
          goto LABEL_59;
        v43 = v87;
        v20 = HIDWORD(v88);
        v23 = (__int64)&v87[HIDWORD(v88)];
        if ( v87 == (__int64 *)v23 )
        {
LABEL_66:
          if ( HIDWORD(v88) < (unsigned int)v88 )
          {
            v20 = (unsigned int)++HIDWORD(v88);
            *(_QWORD *)v23 = v42;
            ++v86;
            goto LABEL_60;
          }
LABEL_59:
          sub_C8CC70((__int64)&v86, *v39, v23, v20, v21, v37);
          if ( (_BYTE)v23 )
          {
LABEL_60:
            v44 = *(_QWORD *)(v42 + 40);
            if ( *v77 == 84 || v72 == v44 )
            {
              v45 = v82;
              if ( HIDWORD(v82) <= (unsigned int)v82 )
              {
                v70 = *(_QWORD *)(v42 + 40);
                v20 = sub_C8D7D0((__int64)&v81, (__int64)&v83, 0, 0x18u, (unsigned __int64 *)&v79, v37);
                v64 = 6LL * (unsigned int)v82;
                v65 = v64 * 4 + v20;
                if ( v64 * 4 + v20 )
                {
                  *(_QWORD *)(v65 + 16) = v42;
                  *(_DWORD *)v65 = v73;
                  *(_QWORD *)(v65 + 8) = v70;
                  v64 = 6LL * (unsigned int)v82;
                }
                v66 = (unsigned __int64)v81;
                v67 = &v81[v64];
                if ( v81 != v67 )
                {
                  v23 = v20;
                  do
                  {
                    if ( v23 )
                    {
                      *(_DWORD *)v23 = *(_DWORD *)v66;
                      *(_QWORD *)(v23 + 8) = *(_QWORD *)(v66 + 8);
                      *(_QWORD *)(v23 + 16) = *(_QWORD *)(v66 + 16);
                    }
                    v66 += 24LL;
                    v23 += 24;
                  }
                  while ( v67 != (int *)v66 );
                  v67 = v81;
                }
                v68 = (int)v79;
                if ( v67 != &v83 )
                {
                  v71 = v20;
                  _libc_free((unsigned __int64)v67);
                  v20 = v71;
                }
                LODWORD(v82) = v82 + 1;
                v81 = (int *)v20;
                HIDWORD(v82) = v68;
              }
              else
              {
                v20 = 3LL * (unsigned int)v82;
                v23 = (__int64)&v81[6 * (unsigned int)v82];
                if ( v23 )
                {
                  *(_QWORD *)(v23 + 16) = v42;
                  *(_QWORD *)(v23 + 8) = v44;
                  *(_DWORD *)v23 = v73;
                  v45 = v82;
                }
                LODWORD(v82) = v45 + 1;
              }
              goto LABEL_57;
            }
          }
        }
        else
        {
          while ( v42 != *v43 )
          {
            if ( (__int64 *)v23 == ++v43 )
              goto LABEL_66;
          }
        }
      }
      if ( !v40 )
      {
        v46 = *(_QWORD *)(*v39 + 8);
        if ( v46 != sub_BCB2A0(*(_QWORD **)(a1 + 3440)) )
          v40 = *v39;
      }
LABEL_57:
      v39 += 4;
      if ( v41 == v39 )
      {
        v15 = (unsigned __int8 *)v40;
        goto LABEL_28;
      }
      continue;
    }
  }
LABEL_29:
  v12 = v74;
  if ( !v74 )
  {
    v61 = *((_QWORD *)a2 + 1);
    if ( v61 != sub_BCB2A0(*(_QWORD **)(a1 + 3440)) || !v15 )
      v15 = a2;
    goto LABEL_102;
  }
LABEL_30:
  v26 = v90;
  v27 = v87;
  if ( !v90 )
  {
    v28 = &v87[(unsigned int)v88];
    if ( v87 == v28 )
    {
LABEL_99:
      _libc_free((unsigned __int64)v87);
      goto LABEL_35;
    }
LABEL_32:
    while ( 1 )
    {
      v29 = *v27;
      v30 = v27;
      if ( (unsigned __int64)*v27 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v28 == ++v27 )
        goto LABEL_34;
    }
    if ( v27 == v28 )
    {
LABEL_34:
      if ( v26 )
        goto LABEL_35;
      goto LABEL_99;
    }
    while ( 1 )
    {
      v47 = *(_BYTE *)(a1 + 696);
      v78 = v29;
      v48 = v47 & 1;
      if ( v48 )
      {
        v49 = a1 + 704;
        v50 = 3;
      }
      else
      {
        v55 = *(_DWORD *)(a1 + 712);
        v49 = *(_QWORD *)(a1 + 704);
        if ( !v55 )
        {
          v56 = *(_DWORD *)(a1 + 696);
          ++*(_QWORD *)(a1 + 688);
          v79 = 0;
          v57 = (v56 >> 1) + 1;
LABEL_83:
          v58 = 3 * v55;
          goto LABEL_84;
        }
        v50 = v55 - 1;
      }
      v51 = v50 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v52 = (__int64 *)(v49 + 16LL * v51);
      v53 = *v52;
      if ( v29 == *v52 )
        goto LABEL_74;
      v59 = 1;
      v60 = 0;
      while ( v53 != -4096 )
      {
        if ( !v60 && v53 == -8192 )
          v60 = v52;
        v51 = v50 & (v59 + v51);
        v52 = (__int64 *)(v49 + 16LL * v51);
        v53 = *v52;
        if ( v29 == *v52 )
          goto LABEL_74;
        ++v59;
      }
      v58 = 12;
      v55 = 4;
      if ( !v60 )
        v60 = v52;
      v56 = *(_DWORD *)(a1 + 696);
      ++*(_QWORD *)(a1 + 688);
      v79 = v60;
      v57 = (v56 >> 1) + 1;
      if ( !(_BYTE)v48 )
      {
        v55 = *(_DWORD *)(a1 + 712);
        goto LABEL_83;
      }
LABEL_84:
      if ( v58 <= 4 * v57 )
      {
        v55 *= 2;
LABEL_90:
        sub_BB64D0(a1 + 688, v55);
        sub_27400A0(a1 + 688, &v78, &v79);
        v29 = v78;
        v56 = *(_DWORD *)(a1 + 696);
        goto LABEL_86;
      }
      if ( v55 - *(_DWORD *)(a1 + 700) - v57 <= v55 >> 3 )
        goto LABEL_90;
LABEL_86:
      *(_DWORD *)(a1 + 696) = (2 * (v56 >> 1) + 2) | v56 & 1;
      v52 = v79;
      if ( *v79 != -4096 )
        --*(_DWORD *)(a1 + 700);
      *v52 = v29;
      *((_DWORD *)v52 + 2) = 0;
LABEL_74:
      *((_DWORD *)v52 + 2) = v12;
      v54 = v30 + 1;
      if ( v30 + 1 != v28 )
      {
        while ( 1 )
        {
          v29 = *v54;
          v30 = v54;
          if ( (unsigned __int64)*v54 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v28 == ++v54 )
            goto LABEL_77;
        }
        if ( v54 != v28 )
          continue;
      }
LABEL_77:
      v26 = v90;
      goto LABEL_34;
    }
  }
  v28 = &v87[HIDWORD(v88)];
  if ( v87 != v28 )
    goto LABEL_32;
LABEL_35:
  if ( v81 != &v83 )
    _libc_free((unsigned __int64)v81);
  return v12;
}
