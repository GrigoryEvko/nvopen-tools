// Function: sub_1B27A40
// Address: 0x1b27a40
//
void __fastcall sub_1B27A40(char *s, _QWORD **a2, __int64 a3, unsigned int a4, __int64 ***a5)
{
  __int64 *v6; // rax
  __int64 *v7; // r15
  size_t v8; // rax
  __int64 v9; // rax
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // r15
  unsigned __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 **v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 **v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rbx
  int v35; // r8d
  int v36; // r9d
  __int64 v37; // rax
  __int64 *v38; // rax
  __int128 v39; // rdi
  __int64 v40; // rcx
  __int64 *v41; // rax
  __int64 v42; // r12
  __int64 v43; // rbx
  _QWORD *v44; // rdi
  __int64 v45; // r12
  __int64 v46; // r13
  _QWORD **v47; // rax
  _QWORD *v48; // rdi
  __int64 v49; // r12
  __int64 v50; // r13
  _QWORD **v51; // rax
  _QWORD *v52; // rdi
  __int64 v58; // [rsp+30h] [rbp-1B0h]
  __int64 v59; // [rsp+40h] [rbp-1A0h]
  __int64 v60; // [rsp+48h] [rbp-198h]
  __int64 v61; // [rsp+58h] [rbp-188h]
  __int64 v62[4]; // [rsp+60h] [rbp-180h] BYREF
  __int64 v63[3]; // [rsp+80h] [rbp-160h] BYREF
  _QWORD *v64; // [rsp+98h] [rbp-148h]
  __int64 v65; // [rsp+A0h] [rbp-140h]
  int v66; // [rsp+A8h] [rbp-138h]
  __int64 v67; // [rsp+B0h] [rbp-130h]
  __int64 v68; // [rsp+B8h] [rbp-128h]
  __int64 *v69; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v70; // [rsp+D8h] [rbp-108h]
  _QWORD **v71; // [rsp+E0h] [rbp-100h] BYREF
  __int64 v72; // [rsp+E8h] [rbp-F8h]
  __int64 v73; // [rsp+F0h] [rbp-F0h]
  _BYTE *v74; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+128h] [rbp-B8h]
  _BYTE v76[176]; // [rsp+130h] [rbp-B0h] BYREF

  v64 = *a2;
  memset(v63, 0, sizeof(v63));
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v6 = (__int64 *)sub_1643270(v64);
  v7 = (__int64 *)sub_16453E0(v6, 0);
  v74 = v76;
  v75 = 0x1000000000LL;
  v8 = strlen(s);
  v9 = sub_16321C0((__int64)a2, (__int64)s, v8, 1);
  v58 = v9;
  if ( v9 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 24LL);
    v13 = v12;
    if ( a5 && *(_DWORD *)(v12 + 12) <= 2u )
    {
      v45 = sub_16471D0(v64, 0);
      v46 = sub_1646BA0(v7, 0);
      v47 = (_QWORD **)sub_1643350(v64);
      v48 = *v47;
      v71 = v47;
      v73 = v45;
      v69 = (__int64 *)&v71;
      v72 = v46;
      v70 = 0x800000003LL;
      v13 = sub_1645600(v48, &v71, 3, 0);
    }
    v14 = *(_QWORD *)(v58 - 24);
    if ( v14 )
    {
      v15 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
      v16 = (unsigned int)(v15 + 1);
      if ( HIDWORD(v75) < (unsigned int)v16 )
        sub_16CD150((__int64)&v74, v76, v16, 8, v10, v11);
      if ( (_DWORD)v15 )
      {
        v17 = v14;
        v18 = 24 * v15;
        v19 = 0;
        v61 = v18;
        do
        {
          if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
            v20 = *(_QWORD *)(v17 - 8);
          else
            v20 = v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
          v21 = *(_QWORD *)(v20 + v19);
          if ( v13 != v12 )
          {
            v22 = (__int64 **)sub_16471D0(v64, 0);
            v59 = sub_15A06D0(v22, 0, v23, v24);
            v60 = sub_15A0A60(v21, 1u);
            v71 = (_QWORD **)sub_15A0A60(v21, 0);
            v73 = v59;
            v69 = (__int64 *)&v71;
            v72 = v60;
            v70 = 0x800000003LL;
            v21 = sub_159F090((__int64 **)v13, (__int64 *)&v71, 3, v25);
          }
          v26 = (unsigned int)v75;
          if ( (unsigned int)v75 >= HIDWORD(v75) )
          {
            sub_16CD150((__int64)&v74, v76, 0, 8, v10, v11);
            v26 = (unsigned int)v75;
          }
          v19 += 24;
          *(_QWORD *)&v74[8 * v26] = v21;
          LODWORD(v75) = v75 + 1;
        }
        while ( v61 != v19 );
      }
    }
    sub_15E55B0(v58);
  }
  else
  {
    v49 = sub_16471D0(v64, 0);
    v50 = sub_1646BA0(v7, 0);
    v51 = (_QWORD **)sub_1643350(v64);
    v52 = *v51;
    v71 = v51;
    v73 = v49;
    v69 = (__int64 *)&v71;
    v72 = v50;
    v70 = 0x800000003LL;
    v13 = sub_1645600(v52, &v71, 3, 0);
  }
  v27 = sub_1643350(v64);
  v62[0] = sub_159C470(v27, a4, 0);
  v62[1] = a3;
  v29 = *(unsigned int *)(v13 + 12);
  if ( (unsigned int)v29 > 2 )
  {
    v30 = (__int64 **)sub_16471D0(v64, 0);
    if ( a5 )
      v33 = sub_15A4A70(a5, (__int64)v30);
    else
      v33 = sub_15A06D0(v30, 0, v31, v32);
    v62[2] = v33;
    v29 = *(unsigned int *)(v13 + 12);
  }
  v34 = sub_159F090((__int64 **)v13, v62, v29, v28);
  v37 = (unsigned int)v75;
  if ( (unsigned int)v75 >= HIDWORD(v75) )
  {
    sub_16CD150((__int64)&v74, v76, 0, 8, v35, v36);
    v37 = (unsigned int)v75;
  }
  *(_QWORD *)&v74[8 * v37] = v34;
  LODWORD(v75) = v75 + 1;
  v38 = sub_1645D80((__int64 *)v13, (unsigned int)v75);
  *((_QWORD *)&v39 + 1) = v74;
  *(_QWORD *)&v39 = v38;
  v41 = (__int64 *)sub_159DFD0(v39, (unsigned int)v75, v40);
  v42 = *v41;
  v43 = (__int64)v41;
  LOWORD(v71) = 257;
  if ( *s )
  {
    v69 = (__int64 *)s;
    LOBYTE(v71) = 3;
  }
  v44 = sub_1648A60(88, 1u);
  if ( v44 )
    sub_15E51E0((__int64)v44, (__int64)a2, v42, 0, 6, v43, (__int64)&v69, 0, 0, 0, 0);
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
  if ( v63[0] )
    sub_161E7C0((__int64)v63, v63[0]);
}
