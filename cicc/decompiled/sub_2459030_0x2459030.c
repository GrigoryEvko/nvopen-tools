// Function: sub_2459030
// Address: 0x2459030
//
void __fastcall sub_2459030(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  char v10; // r8
  _QWORD *v11; // rax
  __int64 v12; // rbx
  unsigned int *v13; // r14
  unsigned int *v14; // r13
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rbx
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // r9
  __int64 v25; // rbx
  unsigned int *v26; // rbx
  unsigned int *v27; // r12
  __int64 v28; // rdx
  unsigned int v29; // esi
  _BYTE *v30; // rax
  __int64 v31; // r15
  __int64 v32; // rax
  _QWORD *v33; // rax
  __int64 v34; // r9
  __int64 v35; // rbx
  unsigned int *v36; // r15
  unsigned int *v37; // r13
  __int64 v38; // rdx
  unsigned int v39; // esi
  _QWORD *v40; // r13
  _QWORD *v41; // rbx
  unsigned __int64 v42; // rsi
  _QWORD *v43; // rax
  _QWORD *v44; // rdi
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // rax
  _QWORD *v48; // rdi
  __int64 v49; // rcx
  __int64 v50; // rdx
  char v51; // al
  __m128i *v52; // rsi
  __int64 v53; // [rsp-10h] [rbp-170h]
  unsigned int v54; // [rsp+8h] [rbp-158h]
  char v56; // [rsp+10h] [rbp-150h]
  _BYTE *v57; // [rsp+18h] [rbp-148h]
  unsigned __int8 v59; // [rsp+20h] [rbp-140h]
  _QWORD v60[4]; // [rsp+40h] [rbp-120h] BYREF
  char v61; // [rsp+60h] [rbp-100h]
  char v62; // [rsp+61h] [rbp-FFh]
  _QWORD v63[4]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v64; // [rsp+90h] [rbp-D0h]
  unsigned int *v65; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+A8h] [rbp-B8h]
  _BYTE v67[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v68; // [rsp+D0h] [rbp-90h]
  __int64 v69; // [rsp+D8h] [rbp-88h]
  __int64 v70; // [rsp+E0h] [rbp-80h]
  __int64 v71; // [rsp+E8h] [rbp-78h]
  void **v72; // [rsp+F0h] [rbp-70h]
  void **v73; // [rsp+F8h] [rbp-68h]
  __int64 v74; // [rsp+100h] [rbp-60h]
  int v75; // [rsp+108h] [rbp-58h]
  __int16 v76; // [rsp+10Ch] [rbp-54h]
  char v77; // [rsp+10Eh] [rbp-52h]
  __int64 v78; // [rsp+110h] [rbp-50h]
  __int64 v79; // [rsp+118h] [rbp-48h]
  void *v80; // [rsp+120h] [rbp-40h] BYREF
  void *v81; // [rsp+128h] [rbp-38h] BYREF

  v2 = (_QWORD *)a2;
  v3 = sub_24584A0((_QWORD ***)a1, a2);
  v4 = sub_BD5C60(a2);
  v76 = 512;
  v71 = v4;
  v72 = &v80;
  v73 = &v81;
  LOWORD(v70) = 0;
  v65 = (unsigned int *)v67;
  v80 = &unk_49DA100;
  v66 = 0x200000000LL;
  v74 = 0;
  v75 = 0;
  v77 = 7;
  v78 = 0;
  v79 = 0;
  v68 = 0;
  v69 = 0;
  v81 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v65, a2);
  if ( *(_BYTE *)(a1 + 10)
    || (_BYTE)qword_4FE7008
    || (v17 = sub_B59BC0(a2), sub_AD7890(v17, a2, v18, v19, v20)) && byte_4FE6E48 )
  {
    v5 = sub_B59CA0(a2);
    v6 = sub_AA4E30(v68);
    v7 = sub_9208B0(v6, *(_QWORD *)(v5 + 8));
    v63[1] = v8;
    v63[0] = (unsigned __int64)(v7 + 7) >> 3;
    v9 = sub_CA1930(v63);
    v10 = -1;
    if ( v9 )
    {
      _BitScanReverse64(&v9, v9);
      v10 = 63 - (v9 ^ 0x3F);
    }
    v59 = v10;
    v64 = 257;
    v11 = sub_BD2C40(80, unk_3F148C0);
    v12 = (__int64)v11;
    if ( v11 )
      sub_B4D750((__int64)v11, 1, v3, v5, v59, 2, 1, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v73 + 2))(v73, v12, v63, v69, v70);
    v13 = v65;
    v14 = &v65[4 * (unsigned int)v66];
    if ( v65 != v14 )
    {
      do
      {
        v15 = *((_QWORD *)v13 + 1);
        v16 = *v13;
        v13 += 4;
        sub_B99FD0(v12, v16, v15);
      }
      while ( v14 != v13 );
    }
  }
  else
  {
    v21 = *(_QWORD *)(sub_B59CA0(a2) + 8);
    v62 = 1;
    v60[0] = "pgocount";
    v61 = 3;
    v22 = sub_AA4E30(v68);
    v64 = 257;
    v54 = (unsigned __int8)sub_AE5020(v22, v21);
    v23 = sub_BD2C40(80, unk_3F10A14);
    v24 = v54;
    v57 = v23;
    if ( v23 )
    {
      sub_B4D190((__int64)v23, v21, v3, (__int64)v63, 0, v54, 0, 0);
      v24 = v53;
    }
    (*((void (__fastcall **)(void **, _BYTE *, _QWORD *, __int64, __int64, __int64))*v73 + 2))(
      v73,
      v57,
      v60,
      v69,
      v70,
      v24);
    v25 = 4LL * (unsigned int)v66;
    if ( v65 != &v65[v25] )
    {
      v26 = &v65[v25];
      v27 = v65;
      do
      {
        v28 = *((_QWORD *)v27 + 1);
        v29 = *v27;
        v27 += 4;
        sub_B99FD0((__int64)v57, v29, v28);
      }
      while ( v26 != v27 );
      v2 = (_QWORD *)a2;
    }
    v64 = 257;
    v30 = (_BYTE *)sub_B59CA0((__int64)v2);
    v31 = sub_929C50(&v65, v57, v30, (__int64)v63, 0, 0);
    v32 = sub_AA4E30(v68);
    v56 = sub_AE5020(v32, *(_QWORD *)(v31 + 8));
    v64 = 257;
    v33 = sub_BD2C40(80, unk_3F10A10);
    v35 = (__int64)v33;
    if ( v33 )
      sub_B4D3C0((__int64)v33, v31, v3, 0, v56, v34, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v73 + 2))(v73, v35, v63, v69, v70);
    v36 = v65;
    v37 = &v65[4 * (unsigned int)v66];
    if ( v65 != v37 )
    {
      do
      {
        v38 = *((_QWORD *)v36 + 1);
        v39 = *v36;
        v36 += 4;
        sub_B99FD0(v35, v39, v38);
      }
      while ( v37 != v36 );
    }
    v60[0] = v35;
    v40 = sub_C52410();
    v41 = v40 + 1;
    v42 = sub_C959E0();
    v43 = (_QWORD *)v40[2];
    if ( v43 )
    {
      v44 = v40 + 1;
      do
      {
        while ( 1 )
        {
          v45 = v43[2];
          v46 = v43[3];
          if ( v42 <= v43[4] )
            break;
          v43 = (_QWORD *)v43[3];
          if ( !v46 )
            goto LABEL_29;
        }
        v44 = v43;
        v43 = (_QWORD *)v43[2];
      }
      while ( v45 );
LABEL_29:
      if ( v41 != v44 && v42 >= v44[4] )
        v41 = v44;
    }
    if ( v41 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_46;
    v47 = v41[7];
    if ( !v47 )
      goto LABEL_46;
    v48 = v41 + 6;
    do
    {
      while ( 1 )
      {
        v49 = *(_QWORD *)(v47 + 16);
        v50 = *(_QWORD *)(v47 + 24);
        if ( *(_DWORD *)(v47 + 32) >= dword_4FE6C08 )
          break;
        v47 = *(_QWORD *)(v47 + 24);
        if ( !v50 )
          goto LABEL_38;
      }
      v48 = (_QWORD *)v47;
      v47 = *(_QWORD *)(v47 + 16);
    }
    while ( v49 );
LABEL_38:
    if ( v41 + 6 == v48 || dword_4FE6C08 < *((_DWORD *)v48 + 8) || (v51 = byte_4FE6C88, *((int *)v48 + 9) <= 0) )
LABEL_46:
      v51 = *(_BYTE *)(a1 + 9);
    if ( v51 )
    {
      v63[0] = v57;
      v52 = *(__m128i **)(a1 + 368);
      if ( v52 == *(__m128i **)(a1 + 376) )
      {
        sub_2453580((unsigned __int64 *)(a1 + 360), v52, v63, v60);
      }
      else
      {
        if ( v52 )
        {
          v52->m128i_i64[0] = (__int64)v57;
          v52->m128i_i64[1] = v60[0];
          v52 = *(__m128i **)(a1 + 368);
        }
        *(_QWORD *)(a1 + 368) = v52 + 1;
      }
    }
  }
  sub_B43D60(v2);
  nullsub_61();
  v80 = &unk_49DA100;
  nullsub_63();
  if ( v65 != (unsigned int *)v67 )
    _libc_free((unsigned __int64)v65);
}
