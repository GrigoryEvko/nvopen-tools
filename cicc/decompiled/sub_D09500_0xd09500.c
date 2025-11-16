// Function: sub_D09500
// Address: 0xd09500
//
__int64 __fastcall sub_D09500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 **a5, __int64 a6)
{
  unsigned int v6; // eax
  unsigned int v7; // r12d
  __int64 v9; // rax
  unsigned __int8 **v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  _BYTE *v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // r9
  unsigned __int8 **v16; // r12
  unsigned __int8 **v17; // r13
  unsigned __int8 *v18; // r15
  char v19; // r14
  unsigned __int8 **v20; // rax
  __int64 v21; // rbx
  unsigned __int8 v22; // al
  __int64 v23; // rdi
  unsigned __int8 *v24; // rax
  unsigned int v25; // eax
  char v26; // r9
  unsigned __int8 **v27; // r10
  unsigned int v28; // r14d
  int v29; // r13d
  __int64 v30; // r15
  char v31; // bl
  unsigned __int8 *v32; // rax
  __int64 v33; // rdi
  unsigned int v34; // eax
  unsigned __int8 *v35; // rax
  unsigned __int8 **v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  unsigned int v39; // r14d
  __int64 v40; // r12
  unsigned __int8 v41; // bl
  int v42; // r15d
  __int64 v43; // r10
  __int64 v44; // rdi
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int8 *v49; // rax
  int v50; // esi
  __int64 v53; // [rsp+20h] [rbp-150h]
  char v56; // [rsp+38h] [rbp-138h]
  unsigned __int8 **v57; // [rsp+38h] [rbp-138h]
  char v58; // [rsp+38h] [rbp-138h]
  unsigned __int8 *v59; // [rsp+40h] [rbp-130h]
  __int64 v60; // [rsp+48h] [rbp-128h]
  unsigned __int8 v61; // [rsp+48h] [rbp-128h]
  _BYTE *v62; // [rsp+50h] [rbp-120h] BYREF
  _BYTE *v63; // [rsp+58h] [rbp-118h]
  _BYTE *v64; // [rsp+60h] [rbp-110h]
  unsigned __int8 **v65; // [rsp+70h] [rbp-100h] BYREF
  __int64 v66; // [rsp+78h] [rbp-F8h]
  _BYTE v67[32]; // [rsp+80h] [rbp-F0h] BYREF
  unsigned __int8 *v68; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v69; // [rsp+A8h] [rbp-C8h]
  __int64 v70; // [rsp+B0h] [rbp-C0h]
  __int64 v71; // [rsp+B8h] [rbp-B8h]
  __int64 v72; // [rsp+C0h] [rbp-B0h]
  __int64 v73; // [rsp+C8h] [rbp-A8h]
  unsigned __int8 *v74; // [rsp+D0h] [rbp-A0h] BYREF
  unsigned __int8 **v75; // [rsp+D8h] [rbp-98h]
  __int64 v76; // [rsp+E0h] [rbp-90h]
  __int64 v77; // [rsp+E8h] [rbp-88h]
  __int64 v78; // [rsp+F0h] [rbp-80h]
  __int64 v79; // [rsp+F8h] [rbp-78h]
  __int64 v80; // [rsp+100h] [rbp-70h] BYREF
  unsigned __int8 **v81; // [rsp+108h] [rbp-68h]
  __int64 v82; // [rsp+110h] [rbp-60h]
  __int64 v83; // [rsp+118h] [rbp-58h]
  _QWORD v84[10]; // [rsp+120h] [rbp-50h] BYREF

  v60 = a2;
  v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( !v6 )
    return 0;
  if ( *(_BYTE *)a4 == 84 && *(_QWORD *)(a4 + 40) == *(_QWORD *)(a2 + 40) )
  {
    LOBYTE(v39) = *(_BYTE *)(a6 + 512);
    if ( !(_BYTE)v39 )
    {
      v58 = 0;
      v40 = 0;
      v41 = 0;
      v42 = 0;
      v68 = 0;
      v53 = 8LL * v6;
      do
      {
        v43 = *(_QWORD *)(a4 - 8);
        v44 = *(_QWORD *)a6;
        v45 = *(_QWORD *)(v60 - 8);
        v46 = 0x1FFFFFFFE0LL;
        if ( (*(_DWORD *)(a4 + 4) & 0x7FFFFFF) != 0 )
        {
          v47 = 0;
          do
          {
            if ( *(_QWORD *)(v45 + 32LL * *(unsigned int *)(v60 + 72) + v40) == *(_QWORD *)(v43
                                                                                          + 32LL
                                                                                          * *(unsigned int *)(a4 + 72)
                                                                                          + 8 * v47) )
            {
              v46 = 32 * v47;
              goto LABEL_84;
            }
            ++v47;
          }
          while ( (*(_DWORD *)(a4 + 4) & 0x7FFFFFF) != (_DWORD)v47 );
          v46 = 0x1FFFFFFFE0LL;
        }
LABEL_84:
        v48 = *(_QWORD *)(v43 + v46);
        v82 = 0;
        v83 = 0;
        v80 = v48;
        v84[0] = 0;
        v81 = a5;
        v84[1] = 0;
        v49 = *(unsigned __int8 **)(v45 + 4 * v40);
        v76 = 0;
        v74 = v49;
        v77 = 0;
        v75 = (unsigned __int8 **)a3;
        v78 = 0;
        v79 = 0;
        v50 = sub_CF4D50(v44, (__int64)&v74, (__int64)&v80, a6, 0);
        if ( v58 )
        {
          LODWORD(v68) = sub_D00090(v41 | ((unsigned __int8)v39 << 8) | (unsigned int)(v42 << 9), v50);
          v41 = (unsigned __int8)v68;
          v42 = (int)v68 >> 9;
          v39 = ((unsigned int)v68 >> 8) & 1;
        }
        else
        {
          LODWORD(v68) = v50;
          v42 = v50 >> 9;
          LOBYTE(v39) = BYTE1(v50) & 1;
          v41 = v50;
          v58 = 1;
        }
        if ( v41 == 1 )
          break;
        v40 += 8;
      }
      while ( v53 != v40 );
      return (v42 << 9) | ((v39 & 1) << 8) | v41;
    }
  }
  v80 = 0;
  v65 = (unsigned __int8 **)v67;
  v66 = 0x400000000LL;
  v81 = (unsigned __int8 **)v84;
  v82 = 4;
  LODWORD(v83) = 0;
  BYTE4(v83) = 1;
  v9 = sub_22077B0(8);
  v56 = 0;
  v13 = (_BYTE *)(v9 + 8);
  v62 = (_BYTE *)v9;
  *(_QWORD *)v9 = a2;
  v64 = (_BYTE *)(v9 + 8);
  v63 = (_BYTE *)(v9 + 8);
  v59 = 0;
  do
  {
    v14 = *((_QWORD *)v13 - 1);
    v13 -= 8;
    v63 = v13;
    if ( (*(_BYTE *)(v14 + 7) & 0x40) != 0 )
    {
      v15 = *(_QWORD *)(v14 - 8);
      v16 = (unsigned __int8 **)(v15 + 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF));
    }
    else
    {
      v16 = (unsigned __int8 **)v14;
      v15 = v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF);
    }
    v17 = (unsigned __int8 **)v15;
    if ( (unsigned __int8 **)v15 != v16 )
    {
      while ( 1 )
      {
        v18 = *v17;
        if ( (unsigned __int8 *)v14 == *v17 )
          goto LABEL_19;
        v19 = qword_4F86888;
        if ( (_BYTE)qword_4F86888 )
        {
          v35 = sub_BD42C0(*v17, a2);
          a2 = 6;
          v18 = v35;
          if ( (unsigned __int8 *)v14 == sub_98ACB0(v35, 6u) )
            goto LABEL_49;
          if ( !BYTE4(v83) )
            goto LABEL_51;
          v36 = v81;
          v11 = HIDWORD(v82);
          v10 = &v81[HIDWORD(v82)];
          if ( v81 != v10 )
          {
            while ( v18 != *v36 )
            {
              if ( v10 == ++v36 )
                goto LABEL_64;
            }
            goto LABEL_19;
          }
LABEL_64:
          if ( HIDWORD(v82) < (unsigned int)v82 )
          {
            v11 = (unsigned int)++HIDWORD(v82);
            *v10 = v18;
            ++v80;
          }
          else
          {
LABEL_51:
            a2 = (__int64)v18;
            sub_C8CC70((__int64)&v80, (__int64)v18, (__int64)v10, v11, v12, v15);
            if ( !(_BYTE)v10 )
              goto LABEL_19;
          }
          if ( *v18 != 84 )
          {
            v74 = 0;
            goto LABEL_45;
          }
          v74 = v18;
          a2 = (__int64)v63;
          if ( v63 == v64 )
          {
            v17 += 4;
            sub_D09370((__int64)&v62, v63, &v74);
            if ( v16 == v17 )
              goto LABEL_20;
          }
          else
          {
            if ( v63 )
            {
              *(_QWORD *)v63 = v18;
              a2 = (__int64)v63;
            }
            a2 += 8;
            v17 += 4;
            v63 = (_BYTE *)a2;
            if ( v16 == v17 )
            {
LABEL_20:
              v13 = v63;
              break;
            }
          }
        }
        else
        {
          if ( *v18 == 84 )
          {
            if ( v59 && v18 != v59 )
            {
              v13 = v62;
              v7 = 1;
              goto LABEL_70;
            }
            v59 = *v17;
          }
          v19 = qword_4F86968;
          if ( (_BYTE)qword_4F86968 && (a2 = 6, (unsigned __int8 *)v60 == sub_98ACB0(v18, 6u)) )
          {
LABEL_49:
            v17 += 4;
            v56 = v19;
            if ( v16 == v17 )
              goto LABEL_20;
          }
          else
          {
            if ( BYTE4(v83) )
            {
              v20 = v81;
              v11 = HIDWORD(v82);
              v10 = &v81[HIDWORD(v82)];
              if ( v81 != v10 )
              {
                while ( v18 != *v20 )
                {
                  if ( v10 == ++v20 )
                    goto LABEL_58;
                }
                goto LABEL_19;
              }
LABEL_58:
              if ( HIDWORD(v82) < (unsigned int)v82 )
              {
                ++HIDWORD(v82);
                *v10 = v18;
                ++v80;
                goto LABEL_45;
              }
            }
            a2 = (__int64)v18;
            sub_C8CC70((__int64)&v80, (__int64)v18, (__int64)v10, v11, v12, v15);
            if ( (_BYTE)v10 )
            {
LABEL_45:
              v37 = (unsigned int)v66;
              v11 = HIDWORD(v66);
              v38 = (unsigned int)v66 + 1LL;
              if ( v38 > HIDWORD(v66) )
              {
                a2 = (__int64)v67;
                sub_C8D5F0((__int64)&v65, v67, v38, 8u, v12, v15);
                v37 = (unsigned int)v66;
              }
              v10 = v65;
              v17 += 4;
              v65[v37] = v18;
              LODWORD(v66) = v66 + 1;
              if ( v16 == v17 )
                goto LABEL_20;
            }
            else
            {
LABEL_19:
              v17 += 4;
              if ( v16 == v17 )
                goto LABEL_20;
            }
          }
        }
      }
    }
  }
  while ( v62 != v13 );
  if ( !v59 || (v7 = 1, (unsigned int)(HIDWORD(v82) - v83) <= 1) )
  {
    v7 = 1;
    if ( (_DWORD)v66 )
    {
      v78 = 0;
      v21 = -1;
      v76 = 0;
      v22 = *(_BYTE *)(a6 + 512);
      v79 = 0;
      *(_BYTE *)(a6 + 512) = 1;
      if ( !v56 )
        v21 = a3;
      v61 = v22;
      v77 = 0;
      v23 = *(_QWORD *)a6;
      v74 = (unsigned __int8 *)a4;
      v75 = a5;
      v24 = *v65;
      v69 = v21;
      v70 = 0;
      v68 = v24;
      v71 = 0;
      v72 = 0;
      v73 = 0;
      v25 = sub_CF4D50(v23, (__int64)&v68, (__int64)&v74, a6, 0);
      v7 = v25;
      v26 = v25;
      if ( (_BYTE)v25 == 1 || (_BYTE)v25 && v56 )
      {
        v7 = 1;
      }
      else
      {
        if ( (_DWORD)v66 != 1 )
        {
          v27 = &v68;
          v28 = 1;
          v29 = v66;
          v30 = v21;
          v31 = v25;
          do
          {
            LOBYTE(v7) = v31;
            v57 = v27;
            v32 = v65[v28];
            v76 = 0;
            v74 = (unsigned __int8 *)a4;
            v33 = *(_QWORD *)a6;
            v78 = 0;
            v75 = a5;
            v77 = 0;
            v79 = 0;
            v68 = v32;
            v69 = v30;
            v70 = 0;
            v71 = 0;
            v72 = 0;
            v73 = 0;
            v34 = sub_CF4D50(v33, (__int64)v27, (__int64)&v74, a6, 0);
            v7 = sub_D00090(v34, v7);
            v31 = v7;
            if ( (_BYTE)v7 == 1 )
              break;
            ++v28;
            v27 = v57;
          }
          while ( v28 != v29 );
          v26 = v7;
        }
        LOBYTE(v7) = v26;
      }
      a2 = v61;
      v13 = v62;
      *(_BYTE *)(a6 + 512) = v61;
    }
  }
LABEL_70:
  if ( v13 )
  {
    a2 = v64 - v13;
    j_j___libc_free_0(v13, v64 - v13);
  }
  if ( !BYTE4(v83) )
    _libc_free(v81, a2);
  if ( v65 != (unsigned __int8 **)v67 )
    _libc_free(v65, a2);
  return v7;
}
