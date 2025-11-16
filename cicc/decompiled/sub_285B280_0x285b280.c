// Function: sub_285B280
// Address: 0x285b280
//
void __fastcall sub_285B280(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v9; // r8
  __int64 v10; // r9
  __int16 v11; // ax
  __int64 v12; // rdx
  __int64 *v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  _BYTE *v22; // rdi
  __int64 v23; // r9
  _BYTE *v24; // r14
  _BYTE *v25; // rdx
  _BYTE *v26; // r10
  _BYTE *v27; // rax
  int v28; // edx
  signed __int64 v29; // r10
  __int64 v30; // r8
  __int64 *v31; // r14
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // r14
  __int64 v36; // rax
  __int64 *v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rdx
  __int64 *v41; // r8
  __int64 *v42; // r14
  __int64 v43; // rbx
  __int64 *v44; // r12
  __int64 v45; // rax
  __int64 *v46; // rax
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rdx
  _BYTE *v50; // [rsp+18h] [rbp-108h]
  __int64 *v51; // [rsp+28h] [rbp-F8h]
  __int64 *v52; // [rsp+28h] [rbp-F8h]
  __int64 *v53; // [rsp+28h] [rbp-F8h]
  __int64 *v54; // [rsp+28h] [rbp-F8h]
  signed __int64 v55; // [rsp+28h] [rbp-F8h]
  __int64 v56; // [rsp+30h] [rbp-F0h]
  __int64 v57; // [rsp+30h] [rbp-F0h]
  __int64 *v58; // [rsp+30h] [rbp-F0h]
  __int64 *v59; // [rsp+30h] [rbp-F0h]
  __int64 v60; // [rsp+30h] [rbp-F0h]
  __int64 *v61; // [rsp+38h] [rbp-E8h]
  __int64 v62; // [rsp+38h] [rbp-E8h]
  int v63; // [rsp+38h] [rbp-E8h]
  __int64 *v64; // [rsp+38h] [rbp-E8h]
  __int64 v65; // [rsp+38h] [rbp-E8h]
  __int64 **v66; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v67; // [rsp+48h] [rbp-D8h]
  __int64 *v68; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v69; // [rsp+58h] [rbp-C8h]
  _BYTE *v70; // [rsp+60h] [rbp-C0h]
  __int64 v71; // [rsp+68h] [rbp-B8h]
  _BYTE dest[32]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 *v73; // [rsp+90h] [rbp-90h] BYREF
  __int64 v74; // [rsp+98h] [rbp-88h]
  _BYTE v75[32]; // [rsp+A0h] [rbp-80h] BYREF
  __int64 *v76; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v77; // [rsp+C8h] [rbp-58h]
  _BYTE v78[80]; // [rsp+D0h] [rbp-50h] BYREF

  while ( !sub_DAEB50((__int64)a5, (__int64)a1, **(_QWORD **)(a2 + 32)) )
  {
    v11 = *((_WORD *)a1 + 12);
    if ( v11 == 5 )
    {
      v12 = a1[5];
      v13 = (__int64 *)a1[4];
      v61 = &v13[v12];
      while ( v61 != v13 )
      {
        v14 = *v13++;
        sub_285B280(v14, a2, a3, a4, a5);
      }
      return;
    }
    if ( v11 != 8 )
      goto LABEL_11;
    if ( sub_D968A0(*(_QWORD *)a1[4]) || a1[5] != 2 )
    {
      v11 = *((_WORD *)a1 + 12);
LABEL_11:
      if ( v11 == 6 && sub_D96960(*(_QWORD *)a1[4]) )
      {
        v22 = dest;
        v24 = (_BYTE *)sub_2850B10(a1[4], a1[5], 1);
        v26 = v25;
        v27 = v25;
        v28 = 0;
        v29 = v26 - v24;
        v70 = dest;
        v71 = 0x400000000LL;
        v30 = v29 >> 3;
        if ( (unsigned __int64)v29 > 0x20 )
        {
          v50 = v27;
          v55 = v29;
          v60 = v29 >> 3;
          v65 = v23;
          sub_C8D5F0(v23, dest, v29 >> 3, 8u, v30, v23);
          v28 = v71;
          v27 = v50;
          v29 = v55;
          LODWORD(v30) = v60;
          v23 = v65;
          v22 = &v70[8 * (unsigned int)v71];
        }
        if ( v27 != v24 )
        {
          v57 = v23;
          v63 = v30;
          memcpy(v22, v24, v29);
          v28 = v71;
          v23 = v57;
          LODWORD(v30) = v63;
        }
        LODWORD(v71) = v28 + v30;
        v31 = sub_DC8BD0(a5, v23, 0, 0);
        v76 = (__int64 *)v78;
        v73 = (__int64 *)v75;
        v74 = 0x400000000LL;
        v77 = 0x400000000LL;
        sub_285B280(v31, a2, &v73, &v76, a5);
        v32 = sub_D95540((__int64)v31);
        v33 = sub_D97090((__int64)a5, v32);
        v34 = sub_AD62B0(v33);
        v64 = sub_DD8400((__int64)a5, v34);
        v58 = &v73[(unsigned int)v74];
        if ( v58 != v73 )
        {
          v35 = v73;
          do
          {
            v36 = *v35;
            v66 = &v68;
            v68 = v64;
            v69 = v36;
            v67 = 0x200000002LL;
            v37 = sub_DC8BD0(a5, (__int64)&v66, 0, 0);
            if ( v66 != &v68 )
            {
              v51 = v37;
              _libc_free((unsigned __int64)v66);
              v37 = v51;
            }
            v40 = *(unsigned int *)(a3 + 8);
            if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
            {
              v54 = v37;
              sub_C8D5F0(a3, (const void *)(a3 + 16), v40 + 1, 8u, v38, v39);
              v40 = *(unsigned int *)(a3 + 8);
              v37 = v54;
            }
            ++v35;
            *(_QWORD *)(*(_QWORD *)a3 + 8 * v40) = v37;
            ++*(_DWORD *)(a3 + 8);
          }
          while ( v58 != v35 );
        }
        v41 = v76;
        v59 = &v76[(unsigned int)v77];
        if ( v59 != v76 )
        {
          v42 = a5;
          v43 = a4;
          v44 = v76;
          do
          {
            v45 = *v44;
            v66 = &v68;
            v68 = v64;
            v69 = v45;
            v67 = 0x200000002LL;
            v46 = sub_DC8BD0(v42, (__int64)&v66, 0, 0);
            if ( v66 != &v68 )
            {
              v52 = v46;
              _libc_free((unsigned __int64)v66);
              v46 = v52;
            }
            v49 = *(unsigned int *)(v43 + 8);
            if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(v43 + 12) )
            {
              v53 = v46;
              sub_C8D5F0(v43, (const void *)(v43 + 16), v49 + 1, 8u, v47, v48);
              v49 = *(unsigned int *)(v43 + 8);
              v46 = v53;
            }
            ++v44;
            *(_QWORD *)(*(_QWORD *)v43 + 8 * v49) = v46;
            ++*(_DWORD *)(v43 + 8);
          }
          while ( v59 != v44 );
          v41 = v76;
        }
        if ( v41 != (__int64 *)v78 )
          _libc_free((unsigned __int64)v41);
        if ( v73 != (__int64 *)v75 )
          _libc_free((unsigned __int64)v73);
        if ( v70 != dest )
          _libc_free((unsigned __int64)v70);
      }
      else
      {
        v20 = *(unsigned int *)(a4 + 8);
        if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          sub_C8D5F0(a4, (const void *)(a4 + 16), v20 + 1, 8u, v9, v10);
          v20 = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v20) = a1;
        ++*(_DWORD *)(a4 + 8);
      }
      return;
    }
    sub_285B280(*(_QWORD *)a1[4], a2, a3, a4, a5);
    v56 = a1[6];
    v62 = sub_D33D80(a1, (__int64)a5, v15, v16, v17);
    v18 = sub_D95540(*(_QWORD *)a1[4]);
    v19 = sub_DA2C50((__int64)a5, v18, 0, 0);
    a1 = sub_DC1960((__int64)a5, (__int64)v19, v62, v56, 0);
  }
  v21 = *(unsigned int *)(a3 + 8);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v21 + 1, 8u, v9, v10);
    v21 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v21) = a1;
  ++*(_DWORD *)(a3 + 8);
}
