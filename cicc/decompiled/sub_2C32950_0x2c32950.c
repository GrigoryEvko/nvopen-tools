// Function: sub_2C32950
// Address: 0x2c32950
//
void __fastcall sub_2C32950(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r8
  unsigned __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // r13
  __int64 v33; // rdi
  __int64 v34; // r14
  int v35; // eax
  __int64 v36; // r15
  int v37; // ebx
  __int64 v38; // r13
  __int64 v39; // r15
  int v40; // r12d
  int v41; // r14d
  unsigned __int64 v42; // rax
  int v43; // r11d
  __int64 v44; // rax
  bool v45; // zf
  char v46; // r13
  int v47; // eax
  __int64 v48; // rcx
  char v49; // di
  __int64 v50; // [rsp+48h] [rbp-648h]
  __int64 v51; // [rsp+58h] [rbp-638h]
  __int64 v52; // [rsp+68h] [rbp-628h]
  __int64 *v53; // [rsp+78h] [rbp-618h] BYREF
  __int64 v54; // [rsp+80h] [rbp-610h] BYREF
  __int64 v55; // [rsp+88h] [rbp-608h] BYREF
  __int64 v56; // [rsp+90h] [rbp-600h] BYREF
  char *v57; // [rsp+98h] [rbp-5F8h]
  __int64 v58; // [rsp+A0h] [rbp-5F0h]
  int v59; // [rsp+A8h] [rbp-5E8h]
  char v60; // [rsp+ACh] [rbp-5E4h]
  char v61; // [rsp+B0h] [rbp-5E0h] BYREF
  _QWORD v62[15]; // [rsp+130h] [rbp-560h] BYREF
  char v63[120]; // [rsp+1A8h] [rbp-4E8h] BYREF
  _QWORD v64[12]; // [rsp+220h] [rbp-470h] BYREF
  __int64 v65; // [rsp+280h] [rbp-410h]
  __int64 v66; // [rsp+288h] [rbp-408h]
  __int16 v67; // [rsp+298h] [rbp-3F8h]
  _QWORD v68[12]; // [rsp+2A0h] [rbp-3F0h] BYREF
  __int64 v69; // [rsp+300h] [rbp-390h]
  __int64 v70; // [rsp+308h] [rbp-388h]
  __int16 v71; // [rsp+318h] [rbp-378h]
  __int16 v72; // [rsp+328h] [rbp-368h]
  _QWORD v73[12]; // [rsp+330h] [rbp-360h] BYREF
  __int64 v74; // [rsp+390h] [rbp-300h]
  __int64 v75; // [rsp+398h] [rbp-2F8h]
  __int16 v76; // [rsp+3A8h] [rbp-2E8h]
  _QWORD v77[15]; // [rsp+3B0h] [rbp-2E0h] BYREF
  __int16 v78; // [rsp+428h] [rbp-268h]
  __int16 v79; // [rsp+438h] [rbp-258h]
  _BYTE v80[120]; // [rsp+440h] [rbp-250h] BYREF
  __int16 v81; // [rsp+4B8h] [rbp-1D8h]
  _BYTE v82[120]; // [rsp+4C0h] [rbp-1D0h] BYREF
  __int16 v83; // [rsp+538h] [rbp-158h]
  __int16 v84; // [rsp+548h] [rbp-148h]
  _BYTE v85[120]; // [rsp+550h] [rbp-140h] BYREF
  __int16 v86; // [rsp+5C8h] [rbp-C8h]
  _BYTE v87[120]; // [rsp+5D0h] [rbp-C0h] BYREF
  __int16 v88; // [rsp+648h] [rbp-48h]
  __int16 v89; // [rsp+658h] [rbp-38h]

  v3 = *a1;
  v57 = &v61;
  v53 = &v56;
  v56 = 0;
  v58 = 16;
  v59 = 0;
  v60 = 1;
  sub_2C2F4B0(v62, v3);
  sub_2C31060((__int64)v80, (__int64)v62, v4, v5, v6, v7);
  sub_2ABCC20(v64, (__int64)v80, v8, v9, v10, v11);
  v67 = v81;
  sub_2ABCC20(v68, (__int64)v82, v12, v13, v14, v15);
  v71 = v83;
  v72 = v84;
  sub_2ABCC20(v73, (__int64)v85, v16, v17, v18, v19);
  v76 = v86;
  sub_2ABCC20(v77, (__int64)v87, v20, v21, v22, v23);
  v26 = v65;
  v78 = v88;
  v79 = v89;
  v27 = v66;
LABEL_2:
  v28 = v74;
  v29 = v75 - v74;
  if ( v27 - v26 != v75 - v74 )
  {
LABEL_3:
    v30 = *(_QWORD *)(v27 - 32);
    v31 = *(_QWORD *)(v30 + 120);
    v52 = v30 + 112;
    if ( v31 == v30 + 112 )
    {
      while ( 1 )
      {
LABEL_22:
        sub_2AD7320((__int64)v64, v28, v26, v29, v24, v25);
        v27 = v66;
        v26 = v65;
        v28 = v69;
        if ( v66 - v65 == v70 - v69 )
        {
          if ( v65 == v66 )
            goto LABEL_2;
          v48 = v65;
          while ( *(_QWORD *)v48 == *(_QWORD *)v28 )
          {
            v49 = *(_BYTE *)(v48 + 24);
            if ( v49 != *(_BYTE *)(v28 + 24)
              || v49 && (*(_QWORD *)(v48 + 8) != *(_QWORD *)(v28 + 8) || *(_QWORD *)(v48 + 16) != *(_QWORD *)(v28 + 16)) )
            {
              break;
            }
            v48 += 32;
            v28 += 32;
            if ( v66 == v48 )
              goto LABEL_2;
          }
        }
        v29 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v66 - 32) + 8LL) - 1;
        if ( (unsigned int)v29 <= 1 )
          goto LABEL_2;
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v31 )
          BUG();
        v35 = *(unsigned __int8 *)(v31 - 16);
        v26 = (unsigned int)(v35 - 19);
        if ( (unsigned __int8)(v35 - 19) > 3u )
          break;
        v32 = *(_QWORD *)(v31 + 72);
        v33 = **(_QWORD **)(v31 + 24);
        v34 = sub_2BF0490(v33);
        if ( v34 && *(_BYTE *)(v31 + 80) )
        {
          v45 = *(_QWORD *)(a2 + 16) == 0;
          v54 = *(_QWORD *)(v32 + 40);
          if ( v45 )
LABEL_54:
            sub_4263D6(v33, v28, v26);
          v28 = (__int64)&v54;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64 *))(a2 + 24))(a2, &v54) )
          {
            v28 = v34;
            sub_2C28390((__int64 *)&v53, v34, v26, v29, v24, v25);
          }
        }
LABEL_7:
        v31 = *(_QWORD *)(v31 + 8);
        if ( v52 == v31 )
          goto LABEL_22;
      }
      if ( (_BYTE)v35 != 5 )
        goto LABEL_7;
      v33 = **(_QWORD **)(v31 + 24);
      v51 = sub_2BF0490(v33);
      if ( !v51 )
        goto LABEL_7;
      v36 = *(_QWORD *)(v31 + 72);
      v25 = *(unsigned int *)(v36 + 24);
      if ( (int)v25 <= 0 )
        goto LABEL_7;
      v50 = v31;
      v37 = 0;
      v38 = v36;
      v39 = a2;
      v40 = 0;
      v41 = v25;
      do
      {
        v26 = *(unsigned int *)(v38 + 32);
        v25 = *(_QWORD *)(v38 + 16);
        v29 = (unsigned int)(v40 + *(_DWORD *)(v38 + 40));
        if ( (_DWORD)v26 )
        {
          v26 = (unsigned int)(v26 - 1);
          v28 = (unsigned int)v26 & (37 * (_DWORD)v29);
          v42 = v25 + 16 * v28;
          v43 = *(_DWORD *)v42;
          if ( (_DWORD)v29 == *(_DWORD *)v42 )
          {
LABEL_16:
            v44 = *(_QWORD *)(v42 + 8);
            if ( v44 )
            {
              v45 = *(_QWORD *)(v39 + 16) == 0;
              v55 = *(_QWORD *)(v44 + 40);
              if ( v45 )
                goto LABEL_54;
              v28 = (__int64)&v55;
              v33 = v39;
              v37 |= (*(__int64 (__fastcall **)(__int64, __int64 *))(v39 + 24))(v39, &v55);
            }
          }
          else
          {
            v47 = 1;
            while ( v43 != 0x7FFFFFFF )
            {
              v33 = (unsigned int)(v47 + 1);
              v28 = (unsigned int)v26 & (v47 + (_DWORD)v28);
              v42 = v25 + 16LL * (unsigned int)v28;
              v43 = *(_DWORD *)v42;
              if ( (_DWORD)v29 == *(_DWORD *)v42 )
                goto LABEL_16;
              v47 = v33;
            }
          }
        }
        ++v40;
      }
      while ( v41 != v40 );
      v46 = v37;
      a2 = v39;
      v31 = v50;
      if ( !v46 )
        goto LABEL_7;
      v28 = v51;
      sub_2C28390((__int64 *)&v53, v51, v26, v29, v24, v25);
      v31 = *(_QWORD *)(v50 + 8);
      if ( v52 == v31 )
        goto LABEL_22;
    }
  }
  while ( v26 != v27 )
  {
    if ( *(_QWORD *)v26 != *(_QWORD *)v28 )
      goto LABEL_3;
    v29 = *(unsigned __int8 *)(v26 + 24);
    if ( (_BYTE)v29 != *(_BYTE *)(v28 + 24)
      || (_BYTE)v29 && (*(_QWORD *)(v26 + 8) != *(_QWORD *)(v28 + 8) || *(_QWORD *)(v26 + 16) != *(_QWORD *)(v28 + 16)) )
    {
      goto LABEL_3;
    }
    v26 += 32;
    v28 += 32;
  }
  sub_2AB1B50((__int64)v77);
  sub_2AB1B50((__int64)v73);
  sub_2AB1B50((__int64)v68);
  sub_2AB1B50((__int64)v64);
  sub_2AB1B50((__int64)v87);
  sub_2AB1B50((__int64)v85);
  sub_2AB1B50((__int64)v82);
  sub_2AB1B50((__int64)v80);
  sub_2AB1B50((__int64)v63);
  sub_2AB1B50((__int64)v62);
  if ( !v60 )
    _libc_free((unsigned __int64)v57);
}
