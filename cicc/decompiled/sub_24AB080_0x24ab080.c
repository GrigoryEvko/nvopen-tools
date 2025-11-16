// Function: sub_24AB080
// Address: 0x24ab080
//
void __fastcall sub_24AB080(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r9
  __int64 v6; // r15
  unsigned __int64 v7; // rax
  int v8; // ecx
  _QWORD *v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  _DWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // r15
  __int64 v18; // r8
  __int64 v19; // rsi
  int v20; // edi
  __int64 v21; // rax
  unsigned int v22; // edi
  _QWORD *v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r11
  __int64 v26; // r15
  unsigned __int64 v27; // r15
  _BYTE *v28; // r12
  __int64 v29; // rdx
  unsigned int v30; // esi
  int v31; // r12d
  __int64 *v32; // rax
  unsigned __int64 v33; // r8
  __int64 v34; // [rsp+0h] [rbp-180h]
  __int64 v35; // [rsp+8h] [rbp-178h]
  int v36; // [rsp+20h] [rbp-160h]
  int v37; // [rsp+24h] [rbp-15Ch]
  __int64 v38[6]; // [rsp+30h] [rbp-150h] BYREF
  char v39[32]; // [rsp+60h] [rbp-120h] BYREF
  __int16 v40; // [rsp+80h] [rbp-100h]
  _QWORD v41[4]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v42; // [rsp+B0h] [rbp-D0h]
  _BYTE *v43; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+C8h] [rbp-B8h]
  _BYTE v45[32]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+F0h] [rbp-90h]
  __int64 v47; // [rsp+F8h] [rbp-88h]
  __int64 v48; // [rsp+100h] [rbp-80h]
  _QWORD *v49; // [rsp+108h] [rbp-78h]
  void **v50; // [rsp+110h] [rbp-70h]
  void **v51; // [rsp+118h] [rbp-68h]
  __int64 v52; // [rsp+120h] [rbp-60h]
  int v53; // [rsp+128h] [rbp-58h]
  __int16 v54; // [rsp+12Ch] [rbp-54h]
  char v55; // [rsp+12Eh] [rbp-52h]
  __int64 v56; // [rsp+130h] [rbp-50h]
  __int64 v57; // [rsp+138h] [rbp-48h]
  void *v58; // [rsp+140h] [rbp-40h] BYREF
  void *v59; // [rsp+148h] [rbp-38h] BYREF

  v51 = &v59;
  v49 = (_QWORD *)sub_BD5C60(a2);
  v50 = &v58;
  v43 = v45;
  v58 = &unk_49DA100;
  v44 = 0x200000000LL;
  v52 = 0;
  v59 = &unk_49DA0B0;
  v3 = *(_QWORD *)(a2 + 40);
  v53 = 0;
  v46 = v3;
  v54 = 512;
  v55 = 7;
  v56 = 0;
  v57 = 0;
  v47 = a2 + 24;
  LOWORD(v48) = 0;
  v4 = *(_QWORD *)sub_B46C60(a2);
  v41[0] = v4;
  if ( v4 && (sub_B96E90((__int64)v41, v4, 1), (v6 = v41[0]) != 0) )
  {
    v7 = (unsigned __int64)v43;
    v8 = v44;
    v9 = &v43[16 * (unsigned int)v44];
    if ( v43 != (_BYTE *)v9 )
    {
      while ( *(_DWORD *)v7 )
      {
        v7 += 16LL;
        if ( v9 == (_QWORD *)v7 )
          goto LABEL_28;
      }
      *(_QWORD *)(v7 + 8) = v41[0];
      goto LABEL_8;
    }
LABEL_28:
    if ( (unsigned int)v44 >= (unsigned __int64)HIDWORD(v44) )
    {
      v33 = (unsigned int)v44 + 1LL;
      if ( HIDWORD(v44) < v33 )
      {
        sub_C8D5F0((__int64)&v43, v45, v33, 0x10u, v33, v5);
        v9 = &v43[16 * (unsigned int)v44];
      }
      *v9 = 0;
      v9[1] = v6;
      v6 = v41[0];
      LODWORD(v44) = v44 + 1;
    }
    else
    {
      if ( v9 )
      {
        *(_DWORD *)v9 = 0;
        v9[1] = v6;
        v8 = v44;
        v6 = v41[0];
      }
      LODWORD(v44) = v8 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v43, 0);
    v6 = v41[0];
  }
  if ( v6 )
LABEL_8:
    sub_B91220((__int64)v41, v6);
  v10 = *(__int64 **)(a1 + 8);
  v40 = 257;
  v38[0] = *v10;
  v38[1] = **(_QWORD **)(a1 + 16);
  v11 = **(unsigned int **)(a1 + 24);
  v12 = sub_BCB2D0(v49);
  v38[2] = sub_ACD640(v12, v11, 0);
  v13 = *(_DWORD **)(a1 + 32);
  LODWORD(v11) = (*v13)++;
  v14 = sub_BCB2D0(v49);
  v38[3] = sub_ACD640(v14, (unsigned int)v11, 0);
  v38[4] = *(_QWORD *)(a2 - 32);
  v15 = 0;
  v16 = **(_QWORD **)a1;
  if ( v16 )
    v15 = *(_QWORD *)(v16 + 24);
  v17 = v57;
  v42 = 257;
  v18 = v56 + 56 * v57;
  if ( v56 == v18 )
  {
    v36 = 6;
    v22 = 6;
  }
  else
  {
    v19 = v56;
    v20 = 0;
    do
    {
      v21 = *(_QWORD *)(v19 + 40) - *(_QWORD *)(v19 + 32);
      v19 += 56;
      v20 += v21 >> 3;
    }
    while ( v18 != v19 );
    v22 = v20 + 6;
    v36 = v22 & 0x7FFFFFF;
  }
  v34 = v56;
  LOBYTE(v37) = 16 * (_DWORD)v57 != 0;
  v23 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v57) << 32) | v22);
  v24 = (__int64)v23;
  if ( v23 )
  {
    v25 = v17;
    v26 = (__int64)v23;
    v35 = v25;
    sub_B44260((__int64)v23, **(_QWORD **)(v15 + 16), 56, v36 | (v37 << 28), 0, 0);
    *(_QWORD *)(v24 + 72) = 0;
    sub_B4A290(v24, v15, v16, v38, 5, (__int64)v41, v34, v35);
  }
  else
  {
    v26 = 0;
  }
  if ( (_BYTE)v54 )
  {
    v32 = (__int64 *)sub_BD5C60(v26);
    *(_QWORD *)(v24 + 72) = sub_A7A090((__int64 *)(v24 + 72), v32, -1, 72);
  }
  if ( (unsigned __int8)sub_920620(v26) )
  {
    v31 = v53;
    if ( v52 )
      sub_B99FD0(v24, 3u, v52);
    sub_B45150(v24, v31);
  }
  (*((void (__fastcall **)(void **, __int64, char *, __int64, __int64))*v51 + 2))(v51, v24, v39, v47, v48);
  v27 = (unsigned __int64)v43;
  v28 = &v43[16 * (unsigned int)v44];
  if ( v43 != v28 )
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
  nullsub_61();
  v58 = &unk_49DA100;
  nullsub_63();
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
}
