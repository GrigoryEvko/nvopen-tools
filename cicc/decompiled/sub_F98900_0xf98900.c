// Function: sub_F98900
// Address: 0xf98900
//
bool __fastcall sub_F98900(__int64 a1, __int64 a2)
{
  __int64 v2; // r10
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // rbx
  int v7; // ecx
  bool result; // al
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned __int64 v17; // r14
  int v18; // esi
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r10
  __int64 v26; // r11
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  unsigned __int64 v40; // rsi
  char v41; // r13
  __int64 *v42; // rsi
  __int64 v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // rdx
  _BYTE *v46; // r13
  __int64 v47; // rbx
  _BYTE *v48; // r12
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 v51; // [rsp+0h] [rbp-1D0h]
  __int64 v52; // [rsp+8h] [rbp-1C8h]
  unsigned __int64 v53; // [rsp+8h] [rbp-1C8h]
  __int64 v54; // [rsp+10h] [rbp-1C0h]
  __int64 v55; // [rsp+18h] [rbp-1B8h]
  __int64 v56; // [rsp+18h] [rbp-1B8h]
  __int64 v57; // [rsp+18h] [rbp-1B8h]
  __int64 v58; // [rsp+18h] [rbp-1B8h]
  __int64 v59; // [rsp+20h] [rbp-1B0h]
  __int64 v60; // [rsp+20h] [rbp-1B0h]
  __int64 v61; // [rsp+20h] [rbp-1B0h]
  __int64 v62; // [rsp+20h] [rbp-1B0h]
  bool v63; // [rsp+4Fh] [rbp-181h]
  __int64 v64; // [rsp+50h] [rbp-180h]
  __int64 v65; // [rsp+58h] [rbp-178h]
  __int64 v66; // [rsp+68h] [rbp-168h] BYREF
  __int64 v67; // [rsp+70h] [rbp-160h] BYREF
  __int64 v68; // [rsp+78h] [rbp-158h] BYREF
  __int64 v69; // [rsp+80h] [rbp-150h] BYREF
  __int64 v70; // [rsp+88h] [rbp-148h] BYREF
  _QWORD v71[4]; // [rsp+90h] [rbp-140h] BYREF
  __int16 v72; // [rsp+B0h] [rbp-120h]
  _BYTE *v73; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v74; // [rsp+C8h] [rbp-108h]
  _BYTE v75[16]; // [rsp+D0h] [rbp-100h] BYREF
  __int16 v76; // [rsp+E0h] [rbp-F0h]
  _BYTE *v77; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+118h] [rbp-B8h]
  _BYTE v79[32]; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v80; // [rsp+140h] [rbp-90h]
  __int64 v81; // [rsp+148h] [rbp-88h]
  __int64 v82; // [rsp+150h] [rbp-80h]
  __int64 v83; // [rsp+158h] [rbp-78h]
  void **v84; // [rsp+160h] [rbp-70h]
  void **v85; // [rsp+168h] [rbp-68h]
  __int64 v86; // [rsp+170h] [rbp-60h]
  int v87; // [rsp+178h] [rbp-58h]
  __int16 v88; // [rsp+17Ch] [rbp-54h]
  char v89; // [rsp+17Eh] [rbp-52h]
  __int64 v90; // [rsp+180h] [rbp-50h]
  __int64 v91; // [rsp+188h] [rbp-48h]
  void *v92; // [rsp+190h] [rbp-40h] BYREF
  void *v93; // [rsp+198h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_QWORD *)(a1 - 32);
  if ( v2 == v3 )
    return 0;
  v4 = *(_QWORD *)(v3 + 56);
  v5 = v3 + 48;
  v6 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v4 )
  {
    if ( v6 != v5 )
    {
      if ( !v6 )
        goto LABEL_62;
      if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 <= 0xA )
        return 0;
    }
    goto LABEL_48;
  }
  if ( v6 == v5 )
    return 0;
  if ( !v6 )
    goto LABEL_62;
  v7 = *(unsigned __int8 *)(v6 - 24);
  if ( (unsigned int)(v7 - 30) > 0xA || v4 != v6 || (_BYTE)v7 != 31 || (*(_DWORD *)(v6 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v10 = *(_QWORD *)(v6 - 88);
  v65 = *(_QWORD *)(v6 - 56);
  v64 = v10;
  result = v3 != v65 && v3 != v10;
  if ( result )
  {
    result = v2 != v65 && v2 != v10;
    if ( result )
    {
      v11 = *(_QWORD *)(v65 + 56);
      if ( !v11 )
        goto LABEL_62;
      if ( *(_BYTE *)(v11 - 24) == 84 )
        return 0;
      v12 = *(_QWORD *)(v10 + 56);
      if ( !v12 )
        goto LABEL_62;
      v13 = *(_QWORD *)(a1 - 64);
      v14 = a1;
      result = *(_BYTE *)(v12 - 24) == 84 || v2 == v13;
      if ( result )
        return 0;
      v15 = *(_QWORD *)(v13 + 56);
      v16 = v13 + 48;
      v17 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v15 )
      {
        if ( v16 == v17 )
          return result;
        if ( !v17 )
          goto LABEL_62;
        v18 = *(unsigned __int8 *)(v17 - 24);
        if ( (unsigned int)(v18 - 30) > 0xA
          || v15 != v17
          || (_BYTE)v18 != 31
          || (*(_DWORD *)(v17 - 20) & 0x7FFFFFF) != 3 )
        {
          return result;
        }
        v19 = *(_QWORD *)(v17 - 56);
        v20 = *(_QWORD *)(v17 - 88);
        if ( v13 != v20 && v13 != v19 )
        {
          v63 = v2 != v20 && v2 != v19;
          if ( v63 )
          {
            v21 = *(_QWORD *)(v19 + 56);
            if ( v21 )
            {
              if ( *(_BYTE *)(v21 - 24) == 84 )
                return result;
              v22 = *(_QWORD *)(v20 + 56);
              if ( v22 )
              {
                if ( *(_BYTE *)(v22 - 24) == 84
                  || *(_QWORD *)(v17 - 120) != *(_QWORD *)(v6 - 120)
                  || v65 != v20
                  || v64 != v19 )
                {
                  return result;
                }
                v55 = *(_QWORD *)(a1 - 64);
                v59 = *(_QWORD *)(a1 + 40);
                v83 = sub_BD5C60(a1);
                v84 = &v92;
                v85 = &v93;
                v77 = v79;
                v92 = &unk_49DA100;
                v78 = 0x200000000LL;
                v88 = 512;
                LOWORD(v82) = 0;
                v93 = &unk_49DA0B0;
                v86 = 0;
                v87 = 0;
                v89 = 7;
                v90 = 0;
                v91 = 0;
                v80 = 0;
                v81 = 0;
                sub_D5F1F0((__int64)&v77, a1);
                v23 = *(_QWORD *)(a1 - 96);
                v72 = 257;
                v54 = v23;
                v52 = *(_QWORD *)(v6 - 120);
                v24 = (*((__int64 (__fastcall **)(void **, __int64, __int64))*v84 + 2))(v84, 30, v23);
                v25 = v59;
                v26 = v55;
                v27 = v24;
                if ( !v24 )
                {
                  v51 = v55;
                  v57 = v59;
                  v76 = 257;
                  v61 = sub_B504D0(30, v54, v52, (__int64)&v73, 0, 0);
                  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v85 + 2))(
                    v85,
                    v61,
                    v71,
                    v81,
                    v82);
                  v27 = v61;
                  v25 = v57;
                  v45 = 16LL * (unsigned int)v78;
                  v26 = v51;
                  if ( v77 != &v77[v45] )
                  {
                    v62 = v57;
                    v58 = v3;
                    v46 = &v77[v45];
                    v53 = v6;
                    v47 = v27;
                    v48 = v77;
                    do
                    {
                      v49 = *((_QWORD *)v48 + 1);
                      v50 = *(_DWORD *)v48;
                      v48 += 16;
                      sub_B99FD0(v47, v50, v49);
                    }
                    while ( v46 != v48 );
                    v27 = v47;
                    v25 = v62;
                    v3 = v58;
                    v26 = v51;
                    v6 = v53;
                    v14 = a1;
                  }
                }
                v56 = v26;
                v60 = v25;
                sub_AC2B30(v14 - 96, v27);
                sub_AA5980(v3, v60, 0);
                sub_AC2B30(v14 - 32, v64);
                sub_AA5980(v56, v60, 0);
                sub_AC2B30(v14 - 64, v65);
                if ( a2 )
                {
                  v73 = v75;
                  v74 = 0x400000000LL;
                  sub_F35FA0((__int64)&v73, v60, v3 | 4, v28, v29, v30);
                  sub_F35FA0((__int64)&v73, v60, v64 & 0xFFFFFFFFFFFFFFFBLL, v31, v32, v33);
                  sub_F35FA0((__int64)&v73, v60, v56 | 4, v34, v35, v36);
                  sub_F35FA0((__int64)&v73, v60, v65 & 0xFFFFFFFFFFFFFFFBLL, v37, v38, v39);
                  v40 = (unsigned __int64)v73;
                  sub_FFB3D0(a2, v73, (unsigned int)v74);
                  if ( v73 != v75 )
                    _libc_free(v73, v40);
                }
                v41 = sub_BC8C50(v14, &v66, &v67);
                if ( !v41 )
                {
                  v67 = 1;
                  v66 = 1;
                }
                if ( (unsigned __int8)sub_BC8C50(v6 - 24, &v68, &v69) )
                {
                  if ( !(unsigned __int8)sub_BC8C50(v17 - 24, &v70, v71) )
                  {
                    v71[0] = 1;
                    v43 = 1;
                    v44 = 1;
                    v70 = 1;
                    goto LABEL_55;
                  }
                }
                else
                {
                  v42 = &v70;
                  v69 = 1;
                  v68 = 1;
                  if ( !(unsigned __int8)sub_BC8C50(v17 - 24, &v70, v71) )
                  {
                    v71[0] = 1;
                    v70 = 1;
                    if ( !v41 )
                    {
LABEL_42:
                      nullsub_61();
                      v92 = &unk_49DA100;
                      nullsub_63();
                      if ( v77 != v79 )
                        _libc_free(v77, v42);
                      return v63;
                    }
                    v43 = 1;
                    v44 = 1;
LABEL_55:
                    v73 = (_BYTE *)(v66 * v69 + v67 * v44);
                    v74 = v43 * v67 + v68 * v66;
                    sub_F8E430((unsigned __int64 *)&v73, 2);
                    v42 = (__int64 *)(unsigned int)v73;
                    sub_F8EA30(v14, (unsigned int)v73, v74);
                    goto LABEL_42;
                  }
                }
                v44 = v70;
                v43 = v71[0];
                goto LABEL_55;
              }
            }
LABEL_62:
            BUG();
          }
        }
        return 0;
      }
      if ( v16 != v17 )
      {
        if ( !v17 )
          goto LABEL_62;
        if ( (unsigned int)*(unsigned __int8 *)(v17 - 24) - 30 <= 0xA )
          return result;
      }
LABEL_48:
      BUG();
    }
  }
  return result;
}
