// Function: sub_F26680
// Address: 0xf26680
//
unsigned __int8 *__fastcall sub_F26680(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 *v6; // r8
  __int64 *v7; // r13
  __int64 *v8; // r15
  __int64 *v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // r8
  unsigned __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v18; // eax
  __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r13
  __int64 *v22; // rax
  unsigned __int64 v23; // rax
  int v24; // edx
  unsigned __int8 *v25; // rax
  unsigned __int8 *v26; // r12
  char v28; // dl
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  char *v33; // rdx
  char *v34; // rcx
  unsigned __int8 **v35; // r13
  unsigned __int8 **v36; // r15
  unsigned __int8 *v37; // r12
  unsigned __int64 v38; // rdx
  int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 *v43; // r15
  __int64 v44; // rdx
  _QWORD *v45; // rax
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // r15
  __int64 v49; // r13
  __int64 v50; // rbx
  __int64 v51; // rdx
  unsigned int v52; // esi
  unsigned __int8 *v53; // r15
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r13
  __int64 v58; // rax
  __int64 v59; // [rsp+10h] [rbp-260h]
  char v60; // [rsp+18h] [rbp-258h]
  unsigned __int8 *v61; // [rsp+20h] [rbp-250h]
  __int64 v62; // [rsp+20h] [rbp-250h]
  _QWORD *v63; // [rsp+20h] [rbp-250h]
  _QWORD *v64; // [rsp+20h] [rbp-250h]
  __int64 *v65; // [rsp+20h] [rbp-250h]
  _QWORD v67[4]; // [rsp+40h] [rbp-230h] BYREF
  __int16 v68; // [rsp+60h] [rbp-210h]
  __int64 v69[4]; // [rsp+70h] [rbp-200h] BYREF
  __int16 v70; // [rsp+90h] [rbp-1E0h]
  __int64 *v71; // [rsp+A0h] [rbp-1D0h] BYREF
  __int64 v72; // [rsp+A8h] [rbp-1C8h]
  _BYTE v73[48]; // [rsp+B0h] [rbp-1C0h] BYREF
  unsigned __int8 **v74; // [rsp+E0h] [rbp-190h] BYREF
  __int64 v75; // [rsp+E8h] [rbp-188h]
  _BYTE v76[48]; // [rsp+F0h] [rbp-180h] BYREF
  __int64 v77; // [rsp+120h] [rbp-150h] BYREF
  __int64 *v78; // [rsp+128h] [rbp-148h]
  __int64 v79; // [rsp+130h] [rbp-140h]
  int v80; // [rsp+138h] [rbp-138h]
  unsigned __int8 v81; // [rsp+13Ch] [rbp-134h]
  char v82; // [rsp+140h] [rbp-130h] BYREF

  v4 = a1;
  v71 = (__int64 *)v73;
  v72 = 0x600000000LL;
  v5 = 4LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
  {
    v6 = *(__int64 **)(a3 - 8);
    v7 = &v6[v5];
  }
  else
  {
    v7 = (__int64 *)a3;
    v6 = (__int64 *)(a3 - v5 * 8);
  }
  if ( v6 == v7 )
    return 0;
  v8 = v6;
  v9 = 0;
  do
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(a3 + 40);
      if ( !(unsigned __int8)sub_B19720(
                               *(_QWORD *)(a1 + 80),
                               v10,
                               *(_QWORD *)(*(_QWORD *)(a3 - 8)
                                         + 32LL * *(unsigned int *)(a3 + 72)
                                         + 8LL * (unsigned int)(((__int64)v8 - *(_QWORD *)(a3 - 8)) >> 5))) )
        break;
      v12 = (unsigned int)v72;
      v13 = *v8;
      v14 = (unsigned int)v72 + 1LL;
      if ( v14 > HIDWORD(v72) )
      {
        v10 = (__int64)v73;
        v62 = *v8;
        sub_C8D5F0((__int64)&v71, v73, v14, 8u, v13, v11);
        v12 = (unsigned int)v72;
        v13 = v62;
      }
      v8 += 4;
      v71[v12] = v13;
      LODWORD(v72) = v72 + 1;
      if ( v7 == v8 )
        goto LABEL_11;
    }
    if ( v9 )
      goto LABEL_27;
    v9 = v8;
    v8 += 4;
  }
  while ( v7 != v8 );
LABEL_11:
  if ( v9 && (_DWORD)v72 )
  {
    v10 = 0;
    v61 = (unsigned __int8 *)*v9;
    v15 = *(_QWORD *)(*(_QWORD *)(a3 - 8)
                    + 32LL * *(unsigned int *)(a3 + 72)
                    + 8LL * (unsigned int)(((__int64)v9 - *(_QWORD *)(a3 - 8)) >> 5));
    v59 = v15;
    v60 = sub_98ED60((unsigned __int8 *)*v9, 0, 0, 0, 0) ^ 1;
    if ( !v60 )
      goto LABEL_14;
    v23 = *(_QWORD *)(v15 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v23 == v15 + 48 )
    {
      v25 = 0;
    }
    else
    {
      if ( !v23 )
        BUG();
      v24 = *(unsigned __int8 *)(v23 - 24);
      v25 = (unsigned __int8 *)(v23 - 24);
      if ( (unsigned int)(v24 - 30) >= 0xB )
        v25 = 0;
    }
    if ( v61 != v25 )
    {
LABEL_14:
      v10 = 1;
      v77 = 0;
      v78 = (__int64 *)&v82;
      v74 = (unsigned __int8 **)v76;
      v75 = 0x600000000LL;
      v18 = v72;
      v79 = 32;
      v80 = 0;
      v81 = 1;
      if ( !(_DWORD)v72 )
      {
LABEL_55:
        if ( v60 )
        {
          v38 = *(_QWORD *)(v59 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v38 == v59 + 48 )
          {
            v40 = 0;
          }
          else
          {
            if ( !v38 )
              BUG();
            v39 = *(unsigned __int8 *)(v38 - 24);
            v40 = 0;
            v41 = v38 - 24;
            if ( (unsigned int)(v39 - 30) < 0xB )
              v40 = v41;
          }
          sub_D5F1F0(*(_QWORD *)(v4 + 32), v40);
          v42 = (__int64)v61;
          v43 = *(__int64 **)(v4 + 32);
          v67[0] = sub_BD5D20((__int64)v61);
          v67[2] = ".fr";
          v70 = 257;
          v68 = 773;
          v67[1] = v44;
          v45 = sub_BD2C40(72, unk_3F10A14);
          if ( v45 )
          {
            v63 = v45;
            sub_B549F0((__int64)v45, v42, (__int64)v69, 0, 0);
            v45 = v63;
          }
          v64 = v45;
          (*(void (__fastcall **)(__int64, _QWORD *, _QWORD *, __int64, __int64))(*(_QWORD *)v43[11] + 16LL))(
            v43[11],
            v45,
            v67,
            v43[7],
            v43[8]);
          v46 = (__int64)v64;
          v47 = 16LL * *((unsigned int *)v43 + 2);
          v48 = *v43;
          v49 = v48 + v47;
          if ( v48 != v49 )
          {
            v65 = v9;
            v50 = v46;
            do
            {
              v51 = *(_QWORD *)(v48 + 8);
              v52 = *(_DWORD *)v48;
              v48 += 16;
              sub_B99FD0(v50, v52, v51);
            }
            while ( v49 != v48 );
            v46 = v50;
            v9 = v65;
          }
          v53 = (unsigned __int8 *)*v9;
          if ( *v9 )
          {
            v54 = v9[1];
            *(_QWORD *)v9[2] = v54;
            if ( v54 )
              *(_QWORD *)(v54 + 16) = v9[2];
          }
          *v9 = v46;
          if ( v46 )
          {
            v55 = *(_QWORD *)(v46 + 16);
            v9[1] = v55;
            if ( v55 )
              *(_QWORD *)(v55 + 16) = v9 + 1;
            v9[2] = v46 + 16;
            *(_QWORD *)(v46 + 16) = v9;
          }
          if ( *v53 > 0x1Cu )
          {
            v56 = *(_QWORD *)(v4 + 40);
            v69[0] = (__int64)v53;
            v57 = v56 + 2096;
            sub_F200C0(v56 + 2096, v69);
            v58 = *((_QWORD *)v53 + 2);
            if ( v58 )
            {
              if ( !*(_QWORD *)(v58 + 8) )
              {
                v69[0] = *(_QWORD *)(v58 + 24);
                sub_F200C0(v57, v69);
              }
            }
          }
        }
        v10 = a2;
        v26 = sub_F162A0(v4, a2, a3);
LABEL_46:
        if ( v74 != (unsigned __int8 **)v76 )
          _libc_free(v74, v10);
        if ( !v81 )
          _libc_free(v78, v10);
        goto LABEL_28;
      }
      while ( 1 )
      {
        v19 = v71;
        v20 = v18;
        v21 = v71[v18 - 1];
        LODWORD(v72) = v18 - 1;
        if ( !(_BYTE)v10 )
          goto LABEL_32;
        v22 = v78;
        v20 = HIDWORD(v79);
        v19 = &v78[HIDWORD(v79)];
        if ( v78 != v19 )
        {
          while ( v21 != *v22 )
          {
            if ( v19 == ++v22 )
              goto LABEL_43;
          }
          goto LABEL_20;
        }
LABEL_43:
        if ( HIDWORD(v79) < (unsigned int)v79 )
        {
          ++HIDWORD(v79);
          *v19 = v21;
          ++v77;
        }
        else
        {
LABEL_32:
          v10 = v21;
          sub_C8CC70((__int64)&v77, v21, (__int64)v19, v20, v16, v17);
          if ( !v28 )
            goto LABEL_20;
        }
        if ( (unsigned int)(HIDWORD(v79) - v80) > 0x20 )
          goto LABEL_45;
        if ( a3 != v21 )
        {
          v10 = 0;
          if ( !sub_98ED60((unsigned __int8 *)v21, 0, 0, 0, 0) )
          {
            if ( *(_BYTE *)v21 <= 0x1Cu || (v10 = 0, sub_98CD60((unsigned __int8 *)v21, 0)) )
            {
LABEL_45:
              v26 = 0;
              goto LABEL_46;
            }
            v31 = (unsigned int)v75;
            v32 = (unsigned int)v75 + 1LL;
            if ( v32 > HIDWORD(v75) )
            {
              sub_C8D5F0((__int64)&v74, v76, v32, 8u, v29, v30);
              v31 = (unsigned int)v75;
            }
            v74[v31] = (unsigned __int8 *)v21;
            LODWORD(v75) = v75 + 1;
            if ( (*(_BYTE *)(v21 + 7) & 0x40) != 0 )
            {
              v33 = *(char **)(v21 - 8);
              v34 = &v33[32 * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF)];
            }
            else
            {
              v34 = (char *)v21;
              v33 = (char *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF));
            }
            v10 = (__int64)&v71[(unsigned int)v72];
            sub_F091F0((__int64)&v71, (char *)v10, v33, v34);
          }
        }
LABEL_20:
        v18 = v72;
        if ( !(_DWORD)v72 )
        {
          v35 = &v74[(unsigned int)v75];
          if ( v35 != v74 )
          {
            v36 = v74;
            do
            {
              v37 = *v36++;
              sub_B44F30(v37);
              sub_B44B50((__int64 *)v37, v10);
              sub_B44A60((__int64)v37);
            }
            while ( v35 != v36 );
            v4 = a1;
          }
          goto LABEL_55;
        }
        v10 = v81;
      }
    }
  }
LABEL_27:
  v26 = 0;
LABEL_28:
  if ( v71 != (__int64 *)v73 )
    _libc_free(v71, v10);
  return v26;
}
