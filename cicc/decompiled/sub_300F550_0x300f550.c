// Function: sub_300F550
// Address: 0x300f550
//
__int64 __fastcall sub_300F550(__int64 a1, __int64 a2)
{
  __int64 *v3; // r12
  __int64 *v4; // rdi
  unsigned int v5; // r12d
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r14
  _QWORD *v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // r13d
  __int64 *v15; // rbx
  int v16; // eax
  int v17; // edi
  unsigned int v18; // r13d
  __int64 *v19; // r14
  __int64 v20; // r8
  __int64 v21; // rsi
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // r12
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 v27; // r13
  _BYTE *v28; // rbx
  __int64 v29; // rdx
  unsigned int v30; // esi
  int v31; // ebx
  void *v32; // r13
  size_t v33; // r12
  int v34; // eax
  _BYTE *v35; // r8
  __int64 v36; // rsi
  _BYTE *v37; // rcx
  __int64 v38; // r12
  __int64 v39; // rax
  unsigned __int64 v40; // rax
  __int64 v41; // r13
  int v42; // eax
  int v43; // ebx
  unsigned __int64 v44; // rdx
  __int64 *v45; // r15
  unsigned int i; // r14d
  _BYTE *v48; // rdi
  __int64 v49; // [rsp+10h] [rbp-1A0h]
  __int64 v50; // [rsp+18h] [rbp-198h]
  _BYTE *v51; // [rsp+28h] [rbp-188h]
  __int64 v52; // [rsp+30h] [rbp-180h]
  int v53; // [rsp+38h] [rbp-178h]
  int v54; // [rsp+38h] [rbp-178h]
  __int64 v56; // [rsp+60h] [rbp-150h]
  __int64 v57; // [rsp+60h] [rbp-150h]
  void *dest; // [rsp+68h] [rbp-148h]
  void *src; // [rsp+70h] [rbp-140h] BYREF
  __int64 v60; // [rsp+78h] [rbp-138h]
  _BYTE v61[32]; // [rsp+80h] [rbp-130h] BYREF
  _BYTE *v62; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v63; // [rsp+A8h] [rbp-108h]
  _BYTE v64[16]; // [rsp+B0h] [rbp-100h] BYREF
  __int16 v65; // [rsp+C0h] [rbp-F0h]
  _BYTE *v66; // [rsp+F0h] [rbp-C0h]
  __int64 v67; // [rsp+F8h] [rbp-B8h]
  _BYTE v68[32]; // [rsp+100h] [rbp-B0h] BYREF
  void *v69; // [rsp+120h] [rbp-90h]
  __int64 v70; // [rsp+128h] [rbp-88h]
  __int64 v71; // [rsp+130h] [rbp-80h]
  __int64 v72; // [rsp+138h] [rbp-78h]
  void **v73; // [rsp+140h] [rbp-70h]
  void **v74; // [rsp+148h] [rbp-68h]
  __int64 v75; // [rsp+150h] [rbp-60h]
  int v76; // [rsp+158h] [rbp-58h]
  __int16 v77; // [rsp+15Ch] [rbp-54h]
  char v78; // [rsp+15Eh] [rbp-52h]
  __int64 v79; // [rsp+160h] [rbp-50h]
  __int64 v80; // [rsp+168h] [rbp-48h]
  void *v81; // [rsp+170h] [rbp-40h] BYREF
  void *v82; // [rsp+178h] [rbp-38h] BYREF

  v3 = *(__int64 **)(a2 + 40);
  v72 = sub_B2BE50(a2);
  v73 = &v81;
  v74 = &v82;
  v66 = v68;
  v81 = &unk_49DA100;
  v67 = 0x200000000LL;
  v4 = v3;
  v5 = 0;
  v77 = 512;
  LOWORD(v71) = 0;
  v75 = 0;
  v76 = 0;
  v78 = 7;
  v79 = 0;
  v80 = 0;
  v69 = 0;
  v70 = 0;
  v82 = &unk_49DA0B0;
  v6 = sub_B6E160(v4, 0x37A9u, 0, 0);
  *(_QWORD *)(a1 + 40) = v6;
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 )
    goto LABEL_45;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 + 24);
      if ( *(_BYTE *)v8 == 85 && a2 == sub_B43CB0(*(_QWORD *)(v7 + 24)) )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        goto LABEL_45;
    }
    v9 = (_QWORD *)(*(_QWORD *)(v8 + 40) + 48LL);
    dest = *(void **)(v8 + 40);
    v56 = (__int64)v9;
    v10 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 == (_QWORD *)v10 )
      goto LABEL_55;
    if ( !v10 )
LABEL_53:
      BUG();
    v11 = v10 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 > 0xA )
    {
LABEL_55:
      HIDWORD(v60) = 4;
      src = v61;
      v16 = 0;
      v53 = 0;
    }
    else
    {
      v14 = sub_B46E30(v11);
      v15 = (__int64 *)v61;
      v53 = v14;
      v60 = 0x400000000LL;
      v16 = 0;
      src = v61;
      if ( (unsigned __int64)v14 > 4 )
      {
        sub_C8D5F0((__int64)&src, v61, v14, 8u, v12, v13);
        v16 = v60;
        v15 = (__int64 *)((char *)src + 8 * (unsigned int)v60);
      }
      v17 = v14;
      if ( v14 )
      {
        v52 = v8;
        v18 = 0;
        v19 = v15;
        do
        {
          if ( v19 )
            *v19 = sub_B46EC0(v11, v18);
          ++v18;
          ++v19;
        }
        while ( v17 != v18 );
        v8 = v52;
        v16 = v60;
      }
    }
    v20 = v49;
    v21 = *(_QWORD *)(v8 + 32);
    v22 = v50;
    LOWORD(v20) = 0;
    LOWORD(v22) = 0;
    v49 = v20;
    LODWORD(v60) = v53 + v16;
    v50 = v22;
    sub_AA5D10((__int64)dest, v21, v22, v56);
    LOWORD(v71) = 0;
    v69 = dest;
    v70 = v56;
    v65 = 257;
    v23 = sub_BD2C40(72, unk_3F148B8);
    v24 = (__int64)v23;
    if ( v23 )
      sub_B4C8A0((__int64)v23, v72, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v74 + 2))(v74, v24, &v62, v70, v71);
    v27 = (unsigned __int64)v66;
    v28 = &v66[16 * (unsigned int)v67];
    if ( v66 != v28 )
    {
      do
      {
        v29 = *(_QWORD *)(v27 + 8);
        v30 = *(_DWORD *)v27;
        v27 += 16LL;
        sub_B99FD0(v24, v30, v29);
      }
      while ( v28 != (_BYTE *)v27 );
    }
    v31 = v60;
    v32 = src;
    v62 = v64;
    v33 = 8LL * (unsigned int)v60;
    v63 = 0x800000000LL;
    v34 = v60;
    if ( (unsigned int)v60 > 8uLL )
    {
      sub_C8D5F0((__int64)&v62, v64, (unsigned int)v60, 8u, v25, v26);
      v48 = &v62[8 * (unsigned int)v63];
    }
    else
    {
      v35 = v64;
      if ( !v33 )
        goto LABEL_23;
      v48 = v64;
    }
    memcpy(v48, v32, v33);
    v35 = v62;
    v34 = v31 + v63;
LABEL_23:
    LODWORD(v36) = v34;
    LODWORD(v63) = v34;
    v57 = v7;
    if ( v34 )
    {
      while ( 1 )
      {
        v37 = &v35[8 * (unsigned int)v36];
        while ( 1 )
        {
          v38 = *((_QWORD *)v37 - 1);
          LODWORD(v36) = v36 - 1;
          LODWORD(v63) = v36;
          v39 = *(_QWORD *)(v38 + 16);
          if ( !v39 )
            break;
          while ( (unsigned __int8)(**(_BYTE **)(v39 + 24) - 30) > 0xAu )
          {
            v39 = *(_QWORD *)(v39 + 8);
            if ( !v39 )
              goto LABEL_29;
          }
          v37 -= 8;
          if ( !(_DWORD)v36 )
            goto LABEL_40;
        }
LABEL_29:
        v40 = *(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v40 != v38 + 48 )
          break;
        if ( (unsigned int)v36 > HIDWORD(v63) )
        {
          v54 = 0;
          v44 = (unsigned int)v36;
          v43 = 0;
          v41 = 0;
LABEL_50:
          sub_C8D5F0((__int64)&v62, v64, v44, 8u, (__int64)v35, v26);
          v35 = v62;
          v36 = (unsigned int)v63;
LABEL_33:
          v45 = (__int64 *)&v35[8 * v36];
          if ( v43 )
          {
            for ( i = 0; i != v43; ++i )
            {
              if ( v45 )
                *v45 = sub_B46EC0(v41, i);
              ++v45;
            }
            LODWORD(v36) = v63 + v54;
            goto LABEL_39;
          }
          LODWORD(v44) = v36;
          goto LABEL_59;
        }
LABEL_39:
        LODWORD(v63) = v36;
        sub_F34560(v38, 0, 0);
        LODWORD(v36) = v63;
        v35 = v62;
        if ( !(_DWORD)v63 )
          goto LABEL_40;
      }
      if ( !v40 )
        goto LABEL_53;
      v41 = v40 - 24;
      v51 = v35;
      if ( (unsigned int)*(unsigned __int8 *)(v40 - 24) - 30 <= 0xA )
      {
        v42 = sub_B46E30(v41);
        v36 = (unsigned int)v36;
        v35 = v51;
        v43 = v42;
        v44 = v42 + (unsigned __int64)(unsigned int)v36;
        v54 = v42;
        if ( v44 > HIDWORD(v63) )
          goto LABEL_50;
        goto LABEL_33;
      }
      v44 = (unsigned int)v36;
      if ( HIDWORD(v63) < (unsigned int)v36 )
      {
        v54 = 0;
        v43 = 0;
        v41 = 0;
        goto LABEL_50;
      }
      v54 = 0;
LABEL_59:
      LODWORD(v36) = v44 + v54;
      goto LABEL_39;
    }
LABEL_40:
    if ( v35 != v64 )
      _libc_free((unsigned __int64)v35);
    if ( src != v61 )
      _libc_free((unsigned __int64)src);
    v7 = *(_QWORD *)(v57 + 8);
    v5 = 1;
  }
  while ( v7 );
LABEL_45:
  nullsub_61();
  v81 = &unk_49DA100;
  nullsub_63();
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  return v5;
}
