// Function: sub_3963E30
// Address: 0x3963e30
//
__int64 __fastcall sub_3963E30(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, char a5)
{
  __int64 v5; // r15
  __int64 v8; // r13
  __int64 v10; // rax
  int v11; // edi
  __int64 v12; // rcx
  int v13; // esi
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rdi
  _QWORD **v17; // rax
  _QWORD *v18; // rax
  int v19; // edx
  char v20; // al
  __int64 v21; // rcx
  _BYTE *v22; // rax
  __int64 v23; // rax
  bool v24; // cc
  __int64 v25; // r14
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  int v29; // r9d
  __int64 v30; // rax
  __int64 *v31; // r13
  __int64 v32; // rax
  __int64 v33; // rax
  _BYTE *v34; // r15
  __int64 v35; // rsi
  __int64 v36; // rdi
  int v37; // ecx
  __int64 *v38; // rax
  unsigned __int64 v39; // r12
  __int64 v40; // rdx
  __int64 *v41; // r14
  __int64 *v42; // r14
  unsigned __int64 v43; // r13
  unsigned __int64 v44; // r13
  unsigned __int64 v45; // rax
  __int64 *v46; // rax
  int v47; // eax
  int v48; // r8d
  __int64 v49; // [rsp+0h] [rbp-3B0h]
  __int64 v50; // [rsp+18h] [rbp-398h]
  __int64 v51; // [rsp+28h] [rbp-388h]
  int v52; // [rsp+30h] [rbp-380h]
  int v53; // [rsp+34h] [rbp-37Ch]
  int v54; // [rsp+48h] [rbp-368h]
  __int64 v58; // [rsp+68h] [rbp-348h] BYREF
  _QWORD v59[6]; // [rsp+70h] [rbp-340h] BYREF
  __int64 v60; // [rsp+A0h] [rbp-310h] BYREF
  _BYTE *v61; // [rsp+A8h] [rbp-308h]
  _BYTE *v62; // [rsp+B0h] [rbp-300h]
  __int64 v63; // [rsp+B8h] [rbp-2F8h]
  int v64; // [rsp+C0h] [rbp-2F0h]
  _BYTE v65[72]; // [rsp+C8h] [rbp-2E8h] BYREF
  __int64 v66; // [rsp+110h] [rbp-2A0h] BYREF
  __int64 v67; // [rsp+118h] [rbp-298h]
  int v68; // [rsp+120h] [rbp-290h]
  __int64 v69; // [rsp+128h] [rbp-288h] BYREF
  __int64 v70; // [rsp+130h] [rbp-280h]
  unsigned __int64 v71; // [rsp+138h] [rbp-278h]
  char v72[64]; // [rsp+150h] [rbp-260h] BYREF
  _BYTE *v73; // [rsp+190h] [rbp-220h] BYREF
  __int64 v74; // [rsp+198h] [rbp-218h]
  _BYTE v75[128]; // [rsp+1A0h] [rbp-210h] BYREF
  __int64 v76; // [rsp+220h] [rbp-190h] BYREF
  __int64 *v77; // [rsp+228h] [rbp-188h]
  __int64 *v78; // [rsp+230h] [rbp-180h]
  __int64 v79; // [rsp+238h] [rbp-178h]
  int v80; // [rsp+240h] [rbp-170h]
  _BYTE v81[136]; // [rsp+248h] [rbp-168h] BYREF
  __int64 v82; // [rsp+2D0h] [rbp-E0h] BYREF
  _BYTE *v83; // [rsp+2D8h] [rbp-D8h]
  _BYTE *v84; // [rsp+2E0h] [rbp-D0h]
  __int64 v85; // [rsp+2E8h] [rbp-C8h]
  int v86; // [rsp+2F0h] [rbp-C0h]
  _BYTE v87[184]; // [rsp+2F8h] [rbp-B8h] BYREF

  v5 = a1;
  v8 = *a3;
  if ( !sub_3961FA0((_BYTE *)a2, (_BYTE *)*a3) || !(unsigned __int8)sub_3963C90(*(_QWORD *)(a2 + 56), v8, a4) )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 32) = a1 + 64;
    *(_QWORD *)(a1 + 40) = a1 + 64;
    *(_QWORD *)(a1 + 48) = 8;
    *(_DWORD *)(a1 + 56) = 0;
    return v5;
  }
  v60 = 0;
  v61 = v65;
  v62 = v65;
  v63 = 8;
  v64 = 0;
  HIDWORD(v51) = 1;
  if ( byte_5055C00 )
  {
    v42 = *(__int64 **)(a2 + 32);
    v43 = sub_1368AA0(v42, a4);
    v44 = v43 / sub_1368DC0((__int64)v42);
    if ( v44 > qword_5055780 )
      v44 = qword_5055780;
    v45 = v44 / (qword_5055780 / (unsigned __int64)qword_50554C0);
    if ( !(_DWORD)v45 )
      LODWORD(v45) = 1;
    HIDWORD(v51) = v45;
  }
  v10 = *(_QWORD *)(a2 + 16);
  v11 = *(_DWORD *)(v10 + 24);
  v52 = v11;
  if ( v11 )
  {
    v12 = *(_QWORD *)(v10 + 8);
    v13 = v11 - 1;
    v14 = (v11 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v15 = (__int64 *)(v12 + 16LL * v14);
    v16 = *v15;
    if ( a4 == *v15 )
    {
LABEL_8:
      v17 = (_QWORD **)v15[1];
      if ( v17 )
      {
        v18 = *v17;
        if ( v18 )
        {
          v19 = 1;
          do
          {
            v18 = (_QWORD *)*v18;
            ++v19;
          }
          while ( v18 );
          v52 = v19;
        }
        else
        {
          v52 = 1;
        }
        goto LABEL_13;
      }
    }
    else
    {
      v47 = 1;
      while ( v16 != -8 )
      {
        v48 = v47 + 1;
        v14 = v13 & (v47 + v14);
        v15 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v15;
        if ( a4 == *v15 )
          goto LABEL_8;
        v47 = v48;
      }
    }
    v52 = 0;
  }
LABEL_13:
  v20 = sub_3960EF0((_BYTE *)*a3);
  v21 = v5 + 64;
  v50 = v5 + 64;
  if ( v20 )
  {
    v22 = (_BYTE *)a3[1];
    *(_DWORD *)(v5 + 8) = 6;
    *(_QWORD *)(v5 + 24) = 0;
    *(_QWORD *)v5 = v22;
    *(_QWORD *)(v5 + 32) = v21;
    *(_DWORD *)(v5 + 12) = HIDWORD(v51);
    *(_QWORD *)(v5 + 40) = v21;
    *(_DWORD *)(v5 + 16) = v52;
    *(_QWORD *)(v5 + 48) = 8;
    *(_DWORD *)(v5 + 56) = 0;
  }
  else
  {
    v76 = 0;
    v23 = *a3;
    v82 = 0;
    v24 = *(_BYTE *)(v23 + 16) <= 0x17u;
    v79 = 16;
    v80 = 0;
    if ( v24 )
      v23 = 0;
    v85 = 16;
    v58 = v23;
    v77 = (__int64 *)v81;
    v78 = (__int64 *)v81;
    v73 = v75;
    v74 = 0x1000000000LL;
    v83 = v87;
    v84 = v87;
    v86 = 0;
    sub_14EF3D0((__int64)&v73, &v58);
    v54 = 0;
    v53 = 0;
    LODWORD(v51) = 0;
    v49 = v5;
    while ( (_DWORD)v74 )
    {
      v25 = *(_QWORD *)&v73[8 * (unsigned int)v74 - 8];
      LODWORD(v74) = v74 - 1;
      sub_165A590((__int64)&v66, (__int64)&v82, v25);
      if ( (_BYTE)v70 )
      {
        LODWORD(v51) = sub_3961DD0(a2, v25, v26, v27, v28, v29) + v51;
        if ( !a5 )
          sub_165A590((__int64)&v66, (__int64)&v60, v25);
        v30 = 3LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v25 + 23) & 0x40) != 0 )
        {
          v31 = *(__int64 **)(v25 - 8);
          v25 = (__int64)&v31[v30];
        }
        else
        {
          v31 = (__int64 *)(v25 - v30 * 8);
        }
        for ( ; (__int64 *)v25 != v31; v31 += 3 )
        {
          v34 = (_BYTE *)*v31;
          v35 = 0;
          v36 = *(_QWORD *)(a2 + 56);
          if ( *(_BYTE *)(*v31 + 16) > 0x17u )
            v35 = *v31;
          v59[0] = v35;
          if ( !(unsigned __int8)sub_3963C90(v36, v35, a4) )
          {
            if ( sub_3961FA0((_BYTE *)a2, v34) )
            {
              if ( v59[0] )
                sub_14EF3D0((__int64)&v73, v59);
            }
            else if ( v59[0] || sub_3953820(*(_QWORD *)(a2 + 56), v34) )
            {
              v32 = sub_1632FA0(*(_QWORD *)(**(_QWORD **)(a2 + 56) + 40LL));
              v33 = sub_3952EB0((__int64)v34, v32);
              v53 += v33;
              v54 += HIDWORD(v33);
              if ( !a5 )
                sub_19E5640((__int64)&v66, (__int64)&v76, (__int64)v34);
            }
          }
        }
      }
    }
    v5 = v49;
    HIDWORD(v66) = *((_DWORD *)a3 + 3) - v54;
    v37 = *((_DWORD *)a3 + 2);
    v67 = v51;
    LODWORD(v66) = v37 - v53;
    v68 = v52;
    sub_16CCCB0(&v69, (__int64)v72, (__int64)&v60);
    if ( (unsigned __int8)sub_3961900((int *)&v66, a2 + 64) == 1 && !a5 )
    {
      v38 = v78;
      v39 = (unsigned __int64)(v78 == v77 ? &v78[HIDWORD(v79)] : &v78[(unsigned int)v79]);
      if ( v78 != (__int64 *)v39 )
      {
        while ( 1 )
        {
          v40 = *v38;
          v41 = v38;
          if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( (__int64 *)v39 == ++v38 )
            goto LABEL_48;
        }
        if ( v38 != (__int64 *)v39 )
        {
          do
          {
            sub_19E5640((__int64)v59, (__int64)(a3 + 16), v40);
            v46 = v41 + 1;
            if ( v41 + 1 == (__int64 *)v39 )
              break;
            while ( 1 )
            {
              v40 = *v46;
              v41 = v46;
              if ( (unsigned __int64)*v46 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( (__int64 *)v39 == ++v46 )
                goto LABEL_48;
            }
          }
          while ( (__int64 *)v39 != v46 );
        }
      }
    }
LABEL_48:
    *(_QWORD *)v49 = v66;
    *(_QWORD *)(v49 + 8) = v67;
    *(_DWORD *)(v49 + 16) = v68;
    sub_16CCEE0((_QWORD *)(v49 + 24), v50, 8, (__int64)&v69);
    if ( v71 != v70 )
      _libc_free(v71);
    if ( v84 != v83 )
      _libc_free((unsigned __int64)v84);
    if ( v73 != v75 )
      _libc_free((unsigned __int64)v73);
    if ( v78 != v77 )
      _libc_free((unsigned __int64)v78);
  }
  if ( v62 != v61 )
    _libc_free((unsigned __int64)v62);
  return v5;
}
