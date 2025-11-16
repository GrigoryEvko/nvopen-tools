// Function: sub_2A30220
// Address: 0x2a30220
//
__int64 __fastcall sub_2A30220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r8d
  __int64 v7; // rbx
  _QWORD *v8; // r12
  _QWORD *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  int v17; // eax
  __int64 *v18; // r15
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // r14
  __int64 *v22; // rax
  int v23; // ecx
  __int64 v24; // r8
  unsigned __int64 *v25; // rdi
  unsigned __int64 *v26; // rdx
  int v27; // esi
  __int64 v28; // rax
  __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // r15
  unsigned __int8 *v32; // r10
  __int64 *v33; // r8
  __int64 v34; // rsi
  __int64 v35; // r14
  _QWORD *v36; // rdi
  unsigned __int64 *v37; // r12
  unsigned __int64 *v38; // r13
  unsigned __int64 v39; // rdi
  __int64 v41; // rsi
  unsigned __int8 *v42; // rsi
  __int64 v43; // [rsp+10h] [rbp-1B0h]
  __int64 v44; // [rsp+18h] [rbp-1A8h]
  __int64 v45; // [rsp+20h] [rbp-1A0h]
  unsigned int v46; // [rsp+44h] [rbp-17Ch]
  __int64 v47; // [rsp+48h] [rbp-178h]
  __int64 *v48; // [rsp+50h] [rbp-170h]
  __int64 v49; // [rsp+58h] [rbp-168h]
  __int64 *v50; // [rsp+58h] [rbp-168h]
  int v51; // [rsp+58h] [rbp-168h]
  __int64 v52; // [rsp+60h] [rbp-160h]
  __int64 v53[4]; // [rsp+80h] [rbp-140h] BYREF
  __int16 v54; // [rsp+A0h] [rbp-120h]
  unsigned __int64 *v55; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v56; // [rsp+B8h] [rbp-108h]
  _BYTE v57[64]; // [rsp+C0h] [rbp-100h] BYREF
  __int64 *v58; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v59; // [rsp+108h] [rbp-B8h]
  _BYTE v60[176]; // [rsp+110h] [rbp-B0h] BYREF

  v6 = 0;
  v7 = *(_QWORD *)(a1 + 80);
  v52 = a1 + 72;
  if ( v7 != a1 + 72 )
  {
    while ( 1 )
    {
      if ( !v7 )
        BUG();
      v8 = (_QWORD *)(*(_QWORD *)(v7 + 24) & 0xFFFFFFFFFFFFFFF8LL);
      if ( v8 == (_QWORD *)(v7 + 24) )
        goto LABEL_59;
      if ( !v8 )
        BUG();
      v9 = v8 - 3;
      if ( (unsigned int)*((unsigned __int8 *)v8 - 24) - 30 > 0xA )
LABEL_59:
        BUG();
      if ( *((_BYTE *)v8 - 24) != 34 )
        goto LABEL_43;
      if ( *((char *)v8 - 17) >= 0 )
        goto LABEL_46;
      v10 = sub_BD2BC0((__int64)(v8 - 3));
      v12 = v10 + v11;
      if ( *((char *)v8 - 17) >= 0 )
        break;
      if ( !(unsigned int)((v12 - sub_BD2BC0((__int64)(v8 - 3))) >> 4) )
        goto LABEL_46;
      if ( *((char *)v8 - 17) >= 0 )
        goto LABEL_56;
      v13 = *(_DWORD *)(sub_BD2BC0((__int64)(v8 - 3)) + 8);
      if ( *((char *)v8 - 17) >= 0 )
        BUG();
      v14 = sub_BD2BC0((__int64)(v8 - 3));
      v16 = -96 - 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
LABEL_13:
      v17 = *((_DWORD *)v8 - 5);
      v18 = (_QWORD *)((char *)v9 + v16);
      v58 = (__int64 *)v60;
      v19 = 32LL * (v17 & 0x7FFFFFF);
      v59 = 0x1000000000LL;
      v20 = v19 + v16;
      v21 = &v9[v19 / 0xFFFFFFFFFFFFFFF8LL];
      v22 = (__int64 *)v60;
      v23 = 0;
      v24 = v20 >> 5;
      if ( (unsigned __int64)v20 > 0x200 )
      {
        v51 = v20 >> 5;
        sub_C8D5F0((__int64)&v58, v60, v20 >> 5, 8u, v24, a6);
        v23 = v59;
        LODWORD(v24) = v51;
        v22 = &v58[(unsigned int)v59];
      }
      if ( v21 != v18 )
      {
        do
        {
          if ( v22 )
            *v22 = *v21;
          v21 += 4;
          ++v22;
        }
        while ( v18 != v21 );
        v23 = v59;
      }
      v55 = (unsigned __int64 *)v57;
      LODWORD(v59) = v23 + v24;
      v56 = 0x100000000LL;
      sub_B56970((__int64)(v8 - 3), (__int64)&v55);
      v54 = 257;
      v48 = v58;
      v47 = (unsigned int)v59;
      v49 = *(v8 - 7);
      v25 = &v55[7 * (unsigned int)v56];
      if ( v55 == v25 )
      {
        v27 = 0;
      }
      else
      {
        v26 = v55;
        v27 = 0;
        do
        {
          v28 = v26[5] - v26[4];
          v26 += 7;
          v27 += v28 >> 3;
        }
        while ( v25 != v26 );
      }
      v29 = (unsigned int)(v59 + v27 + 1);
      v43 = (unsigned int)v56;
      LOBYTE(v21) = 16 * (_DWORD)v56 != 0;
      v44 = (__int64)v55;
      v45 = v8[7];
      v30 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v56) << 32) | v29);
      v31 = (__int64)v30;
      if ( v30 )
      {
        v46 = v46 & 0xE0000000 | v29 & 0x7FFFFFF | ((_DWORD)v21 << 28);
        sub_B44260((__int64)v30, **(_QWORD **)(v45 + 16), 56, v46, (__int64)v8, 0);
        *(_QWORD *)(v31 + 72) = 0;
        sub_B4A290(v31, v45, v49, v48, v47, (__int64)v53, v44, v43);
        v32 = (unsigned __int8 *)v31;
      }
      else
      {
        v32 = 0;
      }
      sub_BD6B90(v32, (unsigned __int8 *)v8 - 24);
      v33 = (__int64 *)(v31 + 48);
      *(_WORD *)(v31 + 2) = *((_WORD *)v8 - 11) & 0xFFC | *(_WORD *)(v31 + 2) & 0xF003;
      *(_QWORD *)(v31 + 72) = v8[6];
      v34 = v8[3];
      v53[0] = v34;
      if ( v34 )
      {
        sub_B96E90((__int64)v53, v34, 1);
        v33 = (__int64 *)(v31 + 48);
        if ( (__int64 *)(v31 + 48) != v53 )
        {
          v41 = *(_QWORD *)(v31 + 48);
          if ( v41 )
          {
LABEL_49:
            v50 = v33;
            sub_B91220((__int64)v33, v41);
            v33 = v50;
          }
          v42 = (unsigned __int8 *)v53[0];
          *(_QWORD *)(v31 + 48) = v53[0];
          if ( v42 )
            sub_B976B0((__int64)v53, v42, (__int64)v33);
          goto LABEL_29;
        }
        if ( v53[0] )
          sub_B91220((__int64)v53, v53[0]);
      }
      else if ( v33 != v53 )
      {
        v41 = *(_QWORD *)(v31 + 48);
        if ( v41 )
          goto LABEL_49;
      }
LABEL_29:
      sub_BD84D0((__int64)(v8 - 3), v31);
      v35 = *(v8 - 15);
      v36 = sub_BD2C40(72, 1u);
      if ( v36 )
        sub_B4C8F0((__int64)v36, v35, 1u, (__int64)v8, 0);
      sub_AA5980(*(v8 - 11), v7 - 24, 0);
      sub_B43D60(v8 - 3);
      v37 = v55;
      v38 = &v55[7 * (unsigned int)v56];
      if ( v55 != v38 )
      {
        do
        {
          v39 = *(v38 - 3);
          v38 -= 7;
          if ( v39 )
            j_j___libc_free_0(v39);
          if ( (unsigned __int64 *)*v38 != v38 + 2 )
            j_j___libc_free_0(*v38);
        }
        while ( v37 != v38 );
        v38 = v55;
      }
      if ( v38 != (unsigned __int64 *)v57 )
        _libc_free((unsigned __int64)v38);
      if ( v58 != (__int64 *)v60 )
        _libc_free((unsigned __int64)v58);
      v6 = 1;
LABEL_43:
      v7 = *(_QWORD *)(v7 + 8);
      if ( v52 == v7 )
        return v6;
    }
    if ( (unsigned int)(v12 >> 4) )
LABEL_56:
      BUG();
LABEL_46:
    v16 = -96;
    goto LABEL_13;
  }
  return v6;
}
