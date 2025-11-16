// Function: sub_DCE310
// Address: 0xdce310
//
__int64 __fastcall sub_DCE310(_QWORD *a1, __int16 a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  __int64 v4; // r13
  __int64 v6; // rsi
  __int64 v8; // rdx
  unsigned int v9; // eax
  char v10; // si
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned int v13; // r15d
  __int64 v14; // r12
  __int64 v15; // rbx
  char *v16; // r8
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v20; // rax
  unsigned int v21; // ebx
  __int64 v22; // rdx
  __int64 v23; // r9
  __int64 v24; // r8
  char v25; // al
  __int64 v26; // rcx
  __int64 v27; // r8
  char v28; // al
  __int64 *v29; // r12
  __int64 v30; // r8
  unsigned __int64 v31; // r15
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 *v34; // rbx
  __int64 *v35; // r12
  __int64 v36; // r9
  unsigned __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // r15
  __int64 *v41; // rdi
  __int64 v42; // rdx
  char *v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rdx
  int v46; // eax
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rdx
  int v50; // eax
  __int64 v51; // rax
  __int64 v52; // rsi
  unsigned __int64 v53; // rcx
  size_t v54; // rdx
  __int64 v55; // rdx
  __int16 v56; // ax
  __int64 v57; // [rsp+10h] [rbp-120h]
  _QWORD *v58; // [rsp+18h] [rbp-118h]
  unsigned __int64 v59; // [rsp+20h] [rbp-110h]
  __int64 v60; // [rsp+20h] [rbp-110h]
  __int64 v61; // [rsp+28h] [rbp-108h]
  void *v62; // [rsp+28h] [rbp-108h]
  __int64 v63; // [rsp+30h] [rbp-100h]
  unsigned int v64; // [rsp+30h] [rbp-100h]
  __int64 v65; // [rsp+30h] [rbp-100h]
  void *dest; // [rsp+38h] [rbp-F8h]
  int desta; // [rsp+38h] [rbp-F8h]
  void *destb; // [rsp+38h] [rbp-F8h]
  __int64 *v69; // [rsp+48h] [rbp-E8h] BYREF
  __int64 *v70; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v71; // [rsp+58h] [rbp-D8h]
  __int64 v72; // [rsp+60h] [rbp-D0h] BYREF
  char *v73; // [rsp+68h] [rbp-C8h]
  __int64 v74; // [rsp+70h] [rbp-C0h]
  int v75; // [rsp+78h] [rbp-B8h]
  char v76; // [rsp+7Ch] [rbp-B4h]
  char v77; // [rsp+80h] [rbp-B0h] BYREF

  v70 = a1;
  LOWORD(v71) = a2;
  if ( a2 != 13 )
    BUG();
  LODWORD(v71) = 720909;
  v4 = a3;
  v6 = *(_QWORD *)a3;
  v8 = *(unsigned int *)(a3 + 8);
  v72 = 0;
  v73 = &v77;
  v74 = 16;
  v75 = 0;
  v76 = 1;
  v9 = sub_DCEB80(&v70, v6, v8, a3);
  v10 = v9;
  if ( (_BYTE)v9 )
  {
    v40 = sub_DCEA30(a1, 13, v4);
    if ( !v76 )
      _libc_free(v73, 13);
    return v40;
  }
  else
  {
    if ( !v76 )
    {
      _libc_free(v73, v9);
      v10 = 0;
    }
    v11 = *(unsigned int *)(v4 + 8);
    v12 = 0;
    v13 = 0;
    if ( !(_DWORD)v11 )
      goto LABEL_14;
    dest = (void *)v3;
    v14 = v4;
    do
    {
      while ( 1 )
      {
        v15 = 8 * v12;
        v16 = (char *)(*(_QWORD *)v14 + v15);
        v17 = *(_QWORD *)v16;
        if ( *(_WORD *)(*(_QWORD *)v16 + 24LL) == 13 )
          break;
        v12 = ++v13;
        if ( (unsigned int)v11 <= v13 )
          goto LABEL_12;
      }
      v18 = *(_QWORD *)v14 + 8 * v11;
      if ( (char *)v18 != v16 + 8 )
      {
        memmove((void *)(*(_QWORD *)v14 + v15), v16 + 8, v18 - (_QWORD)(v16 + 8));
        v16 = (char *)(*(_QWORD *)v14 + v15);
      }
      --*(_DWORD *)(v14 + 8);
      v12 = v13;
      sub_D932D0(v14, v16, *(char **)(v17 + 32), (char *)(*(_QWORD *)(v17 + 32) + 8LL * *(_QWORD *)(v17 + 40)));
      v11 = *(unsigned int *)(v14 + 8);
      v10 = 1;
    }
    while ( (unsigned int)v11 > v13 );
LABEL_12:
    v4 = v14;
    v3 = (unsigned __int64)dest;
    if ( !v10 )
    {
LABEL_14:
      v20 = sub_D95540(**(_QWORD **)v4);
      v58 = sub_DA2C50((__int64)a1, v20, 0, 0);
      desta = *(_DWORD *)(v4 + 8);
      if ( desta != 1 )
      {
        v21 = 1;
        while ( 1 )
        {
          v63 = 8LL * v21;
          if ( (unsigned __int8)sub_DBEBD0((__int64)a1, *(_QWORD *)(*(_QWORD *)v4 + v63)) )
          {
            v24 = v21 - 1;
            v61 = v24;
            v57 = 8 * v24;
            v25 = sub_D9B790(
                    *(_QWORD *)(*(_QWORD *)v4 + 8LL * v21),
                    *(_QWORD *)(*(_QWORD *)v4 + 8 * v24),
                    v22,
                    8 * v24,
                    v24,
                    v23);
            v27 = v21 - 1;
            if ( v25
              || (v3 = v3 & 0xFFFFFF0000000000LL | 0x21,
                  v28 = sub_DCD020(a1, v3, *(_QWORD *)(*(_QWORD *)v4 + 8 * v61), (__int64)v58),
                  v27 = v21 - 1,
                  v28) )
            {
              v42 = *(_QWORD *)(*(_QWORD *)v4 + v57);
              v43 = *(char **)(*(_QWORD *)v4 + 8LL * v21);
              v70 = &v72;
              v72 = v42;
              v73 = v43;
              v71 = 0x600000002LL;
              *(_QWORD *)(*(_QWORD *)v4 + v57) = sub_DCD310(a1, 0xBu, (__int64)&v70, v26, v27);
              v44 = *(_QWORD *)v4 + v63;
              v45 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
              v46 = *(_DWORD *)(v4 + 8);
              if ( v45 != v44 + 8 )
              {
                memmove((void *)v44, (const void *)(v44 + 8), v45 - (v44 + 8));
                v46 = *(_DWORD *)(v4 + 8);
              }
              v39 = 13;
              *(_DWORD *)(v4 + 8) = v46 - 1;
              v47 = sub_DCEA30(a1, 13, v4);
              v41 = v70;
              v40 = v47;
              if ( v70 == &v72 )
                return v40;
LABEL_32:
              _libc_free(v41, v39);
              return v40;
            }
            v59 = v59 & 0xFFFFFF0000000000LL | 0x25;
            if ( (unsigned __int8)sub_DCD020(
                                    a1,
                                    v59,
                                    *(_QWORD *)(*(_QWORD *)v4 + 8 * v61),
                                    *(_QWORD *)(*(_QWORD *)v4 + 8LL * v21)) )
            {
              v48 = *(_QWORD *)v4 + v63;
              v49 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
              v50 = *(_DWORD *)(v4 + 8);
              if ( v49 != v48 + 8 )
              {
                memmove((void *)v48, (const void *)(v48 + 8), v49 - (v48 + 8));
                v50 = *(_DWORD *)(v4 + 8);
              }
              *(_DWORD *)(v4 + 8) = v50 - 1;
              return sub_DCEA30(a1, 13, v4);
            }
          }
          if ( desta == ++v21 )
          {
            desta = *(_DWORD *)(v4 + 8);
            break;
          }
        }
      }
      v29 = *(__int64 **)v4;
      LODWORD(v72) = 13;
      v71 = 0x2000000001LL;
      v70 = &v72;
      v30 = (__int64)&v29[desta];
      if ( v29 != (__int64 *)v30 )
      {
        v31 = *v29;
        v32 = &v72;
        v33 = 1;
        v34 = v29 + 1;
        v35 = &v29[desta];
        v36 = (unsigned int)v31;
        while ( 1 )
        {
          *((_DWORD *)v32 + v33) = v36;
          v37 = HIDWORD(v31);
          LODWORD(v71) = v71 + 1;
          v38 = (unsigned int)v71;
          if ( (unsigned __int64)(unsigned int)v71 + 1 > HIDWORD(v71) )
          {
            sub_C8D5F0((__int64)&v70, &v72, (unsigned int)v71 + 1LL, 4u, v30, v36);
            v38 = (unsigned int)v71;
          }
          *((_DWORD *)v70 + v38) = v37;
          v33 = (unsigned int)(v71 + 1);
          LODWORD(v71) = v71 + 1;
          if ( v35 == v34 )
            break;
          v31 = *v34;
          v36 = (unsigned int)*v34;
          if ( v33 + 1 > (unsigned __int64)HIDWORD(v71) )
          {
            v64 = *v34;
            sub_C8D5F0((__int64)&v70, &v72, v33 + 1, 4u, v30, v36);
            v33 = (unsigned int)v71;
            v36 = v64;
          }
          v32 = v70;
          ++v34;
        }
      }
      v39 = (__int64)&v70;
      v69 = 0;
      v40 = (__int64)sub_C65B40((__int64)(a1 + 129), (__int64)&v70, (__int64 *)&v69, (__int64)off_49DEA80);
      if ( !v40 )
      {
        v51 = a1[133];
        v52 = 8LL * *(unsigned int *)(v4 + 8);
        a1[143] += v52;
        v53 = (v51 + 7) & 0xFFFFFFFFFFFFFFF8LL;
        if ( a1[134] >= v52 + v53 && v51 )
        {
          a1[133] = v52 + v53;
          destb = (void *)((v51 + 7) & 0xFFFFFFFFFFFFFFF8LL);
        }
        else
        {
          destb = (void *)sub_9D1E70((__int64)(a1 + 133), v52, v52, 3);
        }
        v54 = 8LL * *(unsigned int *)(v4 + 8);
        if ( v54 )
          memmove(destb, *(const void **)v4, v54);
        v62 = sub_C65D30((__int64)&v70, a1 + 133);
        v60 = v55;
        v65 = *(unsigned int *)(v4 + 8);
        v40 = sub_A777F0(0x30u, a1 + 133);
        if ( v40 )
        {
          v56 = sub_D95470((__int64 *)destb, v65);
          *(_QWORD *)v40 = 0;
          *(_WORD *)(v40 + 26) = v56;
          *(_QWORD *)(v40 + 16) = v60;
          *(_QWORD *)(v40 + 40) = v65;
          *(_QWORD *)(v40 + 8) = v62;
          *(_WORD *)(v40 + 24) = 13;
          *(_QWORD *)(v40 + 32) = destb;
          *(_WORD *)(v40 + 28) = 6;
        }
        sub_C657C0(a1 + 129, (__int64 *)v40, v69, (__int64)off_49DEA80);
        v39 = v40;
        sub_DAEE00((__int64)a1, v40, *(__int64 **)v4, *(unsigned int *)(v4 + 8));
      }
      v41 = v70;
      if ( v70 == &v72 )
        return v40;
      goto LABEL_32;
    }
    return sub_DCEA30(a1, 13, v4);
  }
}
