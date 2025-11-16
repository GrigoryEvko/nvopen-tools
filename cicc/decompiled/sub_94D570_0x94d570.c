// Function: sub_94D570
// Address: 0x94d570
//
__int64 __fastcall sub_94D570(__int64 a1, __int64 a2, unsigned int a3, unsigned __int64 *a4, char a5, char a6)
{
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int **v13; // r14
  __int64 v14; // rsi
  __m128i *v15; // r15
  unsigned __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v20; // rax
  __int64 v21; // r13
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rsi
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // r10
  __int64 v30; // rdi
  __int64 (__fastcall *v31)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // rcx
  __int64 v35; // r10
  __int64 v36; // rdi
  __int64 (__fastcall *v37)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r14
  unsigned int *v41; // rbx
  unsigned int *v42; // r14
  __int64 v43; // rdx
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // r14
  unsigned int *v47; // rbx
  unsigned int *v48; // r14
  __int64 v49; // rdx
  __int64 v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rax
  __m128i *v53; // [rsp+8h] [rbp-D8h]
  _BYTE *v54; // [rsp+10h] [rbp-D0h]
  _BYTE *v55; // [rsp+10h] [rbp-D0h]
  __int64 v56; // [rsp+18h] [rbp-C8h]
  __int64 v57; // [rsp+18h] [rbp-C8h]
  __m128i *v58; // [rsp+18h] [rbp-C8h]
  _BYTE *v59; // [rsp+18h] [rbp-C8h]
  __int64 v60; // [rsp+18h] [rbp-C8h]
  __int64 v61; // [rsp+18h] [rbp-C8h]
  __int64 v62; // [rsp+18h] [rbp-C8h]
  __int64 v64; // [rsp+20h] [rbp-C0h]
  __int64 v65; // [rsp+20h] [rbp-C0h]
  __int64 v66; // [rsp+20h] [rbp-C0h]
  __int64 v67; // [rsp+20h] [rbp-C0h]
  __int64 v68; // [rsp+20h] [rbp-C0h]
  __int64 v69; // [rsp+28h] [rbp-B8h]
  __int64 v70; // [rsp+28h] [rbp-B8h]
  __int64 v71; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v72; // [rsp+38h] [rbp-A8h]
  __int64 v73; // [rsp+40h] [rbp-A0h]
  _QWORD v74[4]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v75; // [rsp+70h] [rbp-70h]
  _BYTE v76[32]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v77; // [rsp+A0h] [rbp-40h]

  v11 = sub_BCB2D0(*(_QWORD *)(a2 + 40));
  v12 = a3;
  v13 = (unsigned int **)(a2 + 48);
  v69 = sub_ACD640(v11, v12, 0);
  v14 = *(_QWORD *)(a4[9] + 16);
  if ( a6 )
  {
    v56 = *(_QWORD *)(v14 + 16);
    v15 = sub_92F410(a2, v14);
    v53 = sub_92F410(a2, v56);
    v54 = (_BYTE *)sub_AD6530(v53->m128i_i64[1]);
    v57 = sub_90A810(*(__int64 **)(a2 + 32), 10649, 0, 0);
    v71 = (__int64)v15;
    v77 = 257;
    v72 = v69;
    v16 = 0;
    v73 = sub_92B530((unsigned int **)(a2 + 48), 0x21u, (__int64)v53, v54, (__int64)v76);
    v77 = 257;
    if ( v57 )
      v16 = *(_QWORD *)(v57 + 24);
    v17 = sub_921880((unsigned int **)(a2 + 48), v16, v57, (int)&v71, 3, (__int64)v76, 0);
    v77 = 257;
    if ( a5 )
    {
      LODWORD(v74[0]) = 0;
      v18 = sub_94D3D0((unsigned int **)(a2 + 48), v17, (__int64)v74, 1, (__int64)v76);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v18;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    }
    LODWORD(v74[0]) = 1;
    v70 = sub_94D3D0((unsigned int **)(a2 + 48), v17, (__int64)v74, 1, (__int64)v76);
    v35 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *a4, 0, v34);
    v75 = 257;
    if ( v35 != *(_QWORD *)(v70 + 8) )
    {
      v36 = *(_QWORD *)(a2 + 128);
      v37 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v36 + 120LL);
      if ( v37 == sub_920130 )
      {
        if ( *(_BYTE *)v70 > 0x15u )
        {
LABEL_28:
          v61 = v35;
          v77 = 257;
          v39 = sub_BD2C40(72, unk_3F10A14);
          v33 = v39;
          if ( v39 )
            sub_B515B0(v39, v70, v61, v76, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
            *(_QWORD *)(a2 + 136),
            v33,
            v74,
            *(_QWORD *)(a2 + 104),
            *(_QWORD *)(a2 + 112));
          v40 = 4LL * *(unsigned int *)(a2 + 56);
          v41 = *(unsigned int **)(a2 + 48);
          v42 = &v41[v40];
          while ( v42 != v41 )
          {
            v43 = *((_QWORD *)v41 + 1);
            v44 = *v41;
            v41 += 4;
            sub_B99FD0(v33, v44, v43);
          }
          goto LABEL_20;
        }
        v60 = v35;
        if ( (unsigned __int8)sub_AC4810(39) )
          v38 = sub_ADAB70(39, v70, v60, 0);
        else
          v38 = sub_AA93C0(39, v70, v60);
        v35 = v60;
        v33 = v38;
      }
      else
      {
        v62 = v35;
        v52 = v37(v36, 39u, (_BYTE *)v70, v35);
        v35 = v62;
        v33 = v52;
      }
      if ( !v33 )
        goto LABEL_28;
LABEL_20:
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v33;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    }
LABEL_33:
    v33 = v70;
    goto LABEL_20;
  }
  v58 = sub_92F410(a2, v14);
  v20 = sub_AD6530(v58->m128i_i64[1]);
  if ( !a5 )
  {
    v55 = (_BYTE *)v20;
    v25 = sub_90A810(*(__int64 **)(a2 + 32), 10648, 0, 0);
    v77 = 257;
    v65 = v25;
    v26 = 0;
    v72 = sub_92B530((unsigned int **)(a2 + 48), 0x21u, (__int64)v58, v55, (__int64)v76);
    v71 = v69;
    v77 = 257;
    if ( v65 )
      v26 = *(_QWORD *)(v65 + 24);
    v70 = sub_921880((unsigned int **)(a2 + 48), v26, v65, (int)&v71, 2, (__int64)v76, 0);
    v28 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *a4, 0, v27);
    v75 = 257;
    v29 = v28;
    if ( v28 == *(_QWORD *)(v70 + 8) )
      goto LABEL_33;
    v30 = *(_QWORD *)(a2 + 128);
    v31 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v30 + 120LL);
    if ( v31 == sub_920130 )
    {
      if ( *(_BYTE *)v70 > 0x15u )
        goto LABEL_34;
      v66 = v29;
      if ( (unsigned __int8)sub_AC4810(39) )
        v32 = sub_ADAB70(39, v70, v66, 0);
      else
        v32 = sub_AA93C0(39, v70, v66);
      v29 = v66;
      v33 = v32;
    }
    else
    {
      v68 = v29;
      v51 = v31(v30, 39u, (_BYTE *)v70, v29);
      v29 = v68;
      v33 = v51;
    }
    if ( v33 )
      goto LABEL_20;
LABEL_34:
    v67 = v29;
    v77 = 257;
    v45 = sub_BD2C40(72, unk_3F10A14);
    v33 = v45;
    if ( v45 )
      sub_B515B0(v45, v70, v67, v76, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v33,
      v74,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112));
    v46 = 4LL * *(unsigned int *)(a2 + 56);
    v47 = *(unsigned int **)(a2 + 48);
    v48 = &v47[v46];
    while ( v48 != v47 )
    {
      v49 = *((_QWORD *)v47 + 1);
      v50 = *v47;
      v47 += 4;
      sub_B99FD0(v33, v50, v49);
    }
    goto LABEL_20;
  }
  v64 = (__int64)v58;
  v59 = (_BYTE *)v20;
  v21 = sub_90A810(*(__int64 **)(a2 + 32), 10641, 0, 0);
  v77 = 257;
  v74[0] = v69;
  v22 = 0;
  v74[1] = sub_92B530(v13, 0x21u, v64, v59, (__int64)v76);
  v77 = 257;
  if ( v21 )
    v22 = *(_QWORD *)(v21 + 24);
  v23 = sub_921880(v13, v22, v21, (int)v74, 2, (__int64)v76, 0);
  LODWORD(v71) = 0;
  v77 = 257;
  v24 = sub_94D3D0(v13, v23, (__int64)&v71, 1, (__int64)v76);
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v24;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
