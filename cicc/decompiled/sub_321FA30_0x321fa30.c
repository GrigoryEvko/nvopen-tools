// Function: sub_321FA30
// Address: 0x321fa30
//
void __fastcall sub_321FA30(__int64 a1, void (__fastcall ***a2)(_QWORD, _QWORD, _QWORD *), _QWORD *a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r12
  int v13; // r13d
  _BYTE *v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rcx
  char v18; // dl
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  char v23; // dl
  __int64 v24; // rax
  __int64 *v25; // rax
  unsigned __int64 v26; // r13
  _BYTE *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  unsigned __int64 j; // rax
  void (__fastcall *v33)(_QWORD, _QWORD, _QWORD *); // rbx
  __int64 *v34; // rax
  unsigned __int64 v35; // rdx
  char v36; // cl
  int v37; // esi
  int i; // eax
  __int64 v39; // rdi
  __int64 v40; // [rsp+20h] [rbp-2D0h]
  unsigned int v42; // [rsp+3Ch] [rbp-2B4h]
  __int64 v43; // [rsp+40h] [rbp-2B0h]
  __int64 v44; // [rsp+50h] [rbp-2A0h]
  __int64 v45; // [rsp+58h] [rbp-298h]
  void (__fastcall *v46)(_QWORD, _QWORD, _QWORD *); // [rsp+60h] [rbp-290h]
  __int64 v47; // [rsp+70h] [rbp-280h] BYREF
  __int64 v48; // [rsp+78h] [rbp-278h]
  char v49; // [rsp+80h] [rbp-270h]
  char v50; // [rsp+81h] [rbp-26Fh]
  char v51; // [rsp+88h] [rbp-268h]
  unsigned __int16 v52; // [rsp+89h] [rbp-267h]
  __int64 v53[2]; // [rsp+90h] [rbp-260h] BYREF
  _QWORD v54[2]; // [rsp+A0h] [rbp-250h] BYREF
  _QWORD v55[4]; // [rsp+B0h] [rbp-240h] BYREF
  __int16 v56; // [rsp+D0h] [rbp-220h]
  __int64 *v57; // [rsp+E0h] [rbp-210h]
  unsigned __int64 v58; // [rsp+E8h] [rbp-208h]
  _BYTE v59[16]; // [rsp+F0h] [rbp-200h] BYREF
  _BYTE *v60; // [rsp+100h] [rbp-1F0h]
  unsigned __int64 v61; // [rsp+108h] [rbp-1E8h]
  __int64 v62; // [rsp+110h] [rbp-1E0h]
  _BYTE v63[40]; // [rsp+118h] [rbp-1D8h] BYREF
  char v64; // [rsp+140h] [rbp-1B0h]
  unsigned __int64 v65; // [rsp+148h] [rbp-1A8h]
  _BYTE *v66; // [rsp+150h] [rbp-1A0h]
  __int64 v67; // [rsp+158h] [rbp-198h]
  _BYTE v68[48]; // [rsp+160h] [rbp-190h] BYREF
  _BYTE *v69; // [rsp+190h] [rbp-160h]
  __int64 v70; // [rsp+198h] [rbp-158h]
  _BYTE v71[48]; // [rsp+1A0h] [rbp-150h] BYREF
  __int64 *v72; // [rsp+1D0h] [rbp-120h]
  __int64 v73; // [rsp+1D8h] [rbp-118h]
  char v74; // [rsp+1E8h] [rbp-108h]
  _BYTE *v75; // [rsp+1F0h] [rbp-100h]
  __int64 v76; // [rsp+1F8h] [rbp-F8h]
  __int64 v77; // [rsp+200h] [rbp-F0h]
  _BYTE v78[56]; // [rsp+208h] [rbp-E8h] BYREF
  char *v79; // [rsp+240h] [rbp-B0h]
  __int64 v80; // [rsp+248h] [rbp-A8h]
  char v81; // [rsp+250h] [rbp-A0h] BYREF
  char *v82; // [rsp+280h] [rbp-70h]
  __int64 v83; // [rsp+288h] [rbp-68h]
  char v84; // [rsp+290h] [rbp-60h] BYREF

  v7 = (__int64)a3 - *(_QWORD *)(a1 + 1432);
  v8 = a3[3];
  v9 = *(_QWORD *)(a1 + 2752);
  if ( (v7 >> 5) + 1 == *(_DWORD *)(a1 + 1440) )
    v10 = (*(_QWORD *)(a1 + 2760) - v9) >> 5;
  else
    v10 = a3[7];
  v11 = *(_QWORD *)(a1 + 8);
  v12 = v9 + 32 * v8;
  v44 = v12 + 32 * (v10 - v8);
  v13 = *(_DWORD *)(*(_QWORD *)(v11 + 208) + 8LL);
  v14 = (_BYTE *)sub_31DA930(v11);
  v17 = *(_QWORD *)(a1 + 2472);
  v18 = *v14 ^ 1;
  v19 = a3[2];
  if ( (((__int64)a3 - *(_QWORD *)(a1 + 1432)) >> 5) + 1 == *(_DWORD *)(a1 + 1440) )
  {
    v39 = *(_QWORD *)(a1 + 2480) - v19;
    v21 = v17 + v19;
    v40 = v39;
  }
  else
  {
    v20 = a3[6] - v19;
    v21 = v17 + v19;
    v40 = v20;
  }
  v43 = v21;
  v22 = *(_QWORD *)(a1 + 8);
  v49 = v18;
  v50 = v13;
  v23 = 1;
  v24 = *(_QWORD *)(v22 + 216);
  v47 = v43;
  v48 = v40;
  LOBYTE(v24) = *(_BYTE *)(v24 + 1906);
  v51 = v13;
  HIBYTE(v52) = 1;
  LOBYTE(v52) = v24;
  v60 = v63;
  v66 = v68;
  v67 = 0x600000000LL;
  v70 = 0x600000000LL;
  v25 = &v47;
  v57 = &v47;
  v58 = 0;
  v59[8] = 0;
  v61 = 0;
  v62 = 40;
  v64 = 0;
  v69 = v71;
  if ( v40 )
  {
    v23 = sub_124AC50((__int64)v59, v13, 0, v52, v15, v16, v43) ^ 1;
    v40 = v48;
    v25 = v57;
  }
  v64 = v23;
  v75 = v78;
  v26 = 0;
  v72 = &v47;
  v79 = &v81;
  v73 = v40;
  v74 = 0;
  v76 = 0;
  v77 = 40;
  v80 = 0x600000000LL;
  v82 = &v84;
  v83 = 0x600000000LL;
  v78[40] = 1;
  if ( v25 == &v47 )
    goto LABEL_30;
  do
  {
    do
    {
      v46 = **a2;
      v53[0] = (__int64)v54;
      if ( v12 == v44 )
      {
        sub_3219430(v53, byte_3F871B3, (__int64)byte_3F871B3);
      }
      else
      {
        v27 = *(_BYTE **)v12;
        v28 = *(_QWORD *)(v12 + 8);
        v12 += 32;
        sub_3218AB0(v53, v27, (__int64)&v27[v28]);
      }
      v55[0] = v53;
      v56 = 260;
      v46(a2, v59[0], v55);
      if ( (_QWORD *)v53[0] != v54 )
        j_j___libc_free_0(v53[0]);
      ++v26;
      if ( v61 )
      {
        v42 = 0;
        v31 = 0;
        do
        {
          v45 = 8 * v31;
          if ( v60[v31] == 8 )
          {
            v37 = ((__int64 (__fastcall *)(void (__fastcall ***)(_QWORD, _QWORD, _QWORD *), _QWORD))(*a2)[3])(
                    a2,
                    *(_QWORD *)(*(_QWORD *)(a4 + 760) + 16LL * *(_QWORD *)&v66[8 * v31] + 8));
            if ( v37 )
            {
              for ( i = 0; i != v37; ++i )
              {
                if ( v44 != v12 )
                  v12 += 32;
              }
            }
            v26 = *(_QWORD *)&v69[v45];
          }
          else
          {
            for ( j = *(_QWORD *)&v69[8 * v31]; v26 < j; j = *(_QWORD *)&v69[v45] )
            {
              v33 = **a2;
              v53[0] = (__int64)v54;
              if ( v44 == v12 )
              {
                sub_3219430(v53, byte_3F871B3, (__int64)byte_3F871B3);
              }
              else
              {
                sub_3218AB0(v53, *(_BYTE **)v12, *(_QWORD *)v12 + *(_QWORD *)(v12 + 8));
                v12 += 32;
              }
              v55[0] = v53;
              v56 = 260;
              v33(a2, *(unsigned __int8 *)(v43 + v26), v55);
              if ( (_QWORD *)v53[0] != v54 )
                j_j___libc_free_0(v53[0]);
              ++v26;
            }
            v26 = j;
          }
          v31 = ++v42;
        }
        while ( v42 < v61 );
      }
      v34 = v57;
      if ( v64 )
        v35 = v57[1];
      else
        v35 = v65;
      v58 = v35;
      v36 = 1;
      if ( v35 < v57[1] )
      {
        v36 = sub_124AC50(
                (__int64)v59,
                *((_BYTE *)v57 + 24),
                v35,
                *(unsigned __int16 *)((char *)v57 + 25),
                v29,
                v30,
                *v57)
            ^ 1;
        v34 = v57;
      }
      v64 = v36;
    }
    while ( v34 != &v47 );
LABEL_30:
    ;
  }
  while ( v40 != v58 );
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  if ( v60 != v63 )
    _libc_free((unsigned __int64)v60);
}
