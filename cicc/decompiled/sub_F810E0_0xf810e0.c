// Function: sub_F810E0
// Address: 0xf810e0
//
__int64 __fastcall sub_F810E0(__int64 *a1, unsigned int a2, _BYTE *a3, _BYTE *a4, char a5, char a6)
{
  __int64 *v8; // rbx
  __int64 v9; // r15
  __int64 v11; // r8
  _QWORD *v12; // r10
  _QWORD *v13; // rsi
  __int64 v14; // r11
  int v15; // r15d
  __int64 *v16; // r9
  _QWORD *v17; // r14
  unsigned __int64 v18; // rdx
  int v19; // ecx
  _QWORD *v20; // rax
  __int64 v21; // rdi
  char v22; // al
  char v23; // al
  bool v24; // al
  __int64 v25; // rax
  __int64 v26; // rsi
  __int16 v27; // ax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  __int64 v32; // r13
  __int64 v33; // rbx
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 *v36; // r13
  __int64 v37; // r15
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rax
  unsigned __int64 v41; // rsi
  int v42; // eax
  __int64 v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rsi
  int v47; // eax
  int v48; // edi
  __int64 v49; // rdx
  __int64 *v50; // rax
  __int64 v51; // r8
  int v52; // eax
  int v53; // r9d
  __int64 v54; // rsi
  unsigned __int8 *v55; // rsi
  char v56; // [rsp+Bh] [rbp-125h]
  int v57; // [rsp+Ch] [rbp-124h]
  __int64 *v58; // [rsp+10h] [rbp-120h]
  __int64 *v59; // [rsp+18h] [rbp-118h]
  __int64 v60; // [rsp+20h] [rbp-110h]
  __int64 v61; // [rsp+28h] [rbp-108h]
  _QWORD *v62; // [rsp+28h] [rbp-108h]
  _QWORD *v63; // [rsp+30h] [rbp-100h]
  __int64 v67; // [rsp+58h] [rbp-D8h] BYREF
  char v68[32]; // [rsp+60h] [rbp-D0h] BYREF
  __int16 v69; // [rsp+80h] [rbp-B0h]
  __int64 v70[4]; // [rsp+90h] [rbp-A0h] BYREF
  __int16 v71; // [rsp+B0h] [rbp-80h]
  __int64 *v72; // [rsp+C0h] [rbp-70h] BYREF
  _QWORD v73[4]; // [rsp+C8h] [rbp-68h] BYREF
  __int16 v74; // [rsp+E8h] [rbp-48h]
  _QWORD v75[8]; // [rsp+F0h] [rbp-40h] BYREF

  v8 = a1;
  if ( *a3 <= 0x15u && *a4 <= 0x15u )
  {
    v9 = sub_96E6C0(a2, (__int64)a3, a4, a1[1]);
    if ( v9 )
      return v9;
  }
  v11 = a1[71];
  v12 = (_QWORD *)a1[72];
  v13 = *(_QWORD **)(v11 + 56);
  if ( v12 == v13 )
  {
    if ( !v12 )
      BUG();
  }
  else
  {
    v14 = 0x40540000000000LL;
    v15 = 6;
    v16 = a1;
    v17 = (_QWORD *)(*v12 & 0xFFFFFFFFFFFFFFF8LL);
    do
    {
      if ( !v17 )
        BUG();
      v18 = *((unsigned __int8 *)v17 - 24);
      if ( (_BYTE)v18 == 85 )
      {
        v25 = *(v17 - 7);
        if ( v25 )
        {
          if ( !*(_BYTE *)v25 && *(_QWORD *)(v25 + 24) == v17[7] && (*(_BYTE *)(v25 + 33) & 0x20) != 0 )
            v15 += (unsigned int)(*(_DWORD *)(v25 + 36) - 68) < 4;
        }
      }
      v19 = (unsigned __int8)v18;
      if ( a2 == (unsigned __int8)v18 - 29 )
      {
        v20 = (*((_BYTE *)v17 - 17) & 0x40) != 0
            ? (_QWORD *)*(v17 - 4)
            : &v17[-4 * (*((_DWORD *)v17 - 5) & 0x7FFFFFF) - 3];
        if ( a3 == (_BYTE *)*v20 && a4 == (_BYTE *)v20[4] )
        {
          v21 = (__int64)(v17 - 3);
          if ( (unsigned __int8)v18 > 0x36u )
            goto LABEL_21;
          if ( !_bittest64(&v14, v18) )
            goto LABEL_21;
          v59 = v16;
          v61 = v11;
          v63 = v12;
          v56 = *((_BYTE *)v17 - 24);
          v57 = (unsigned __int8)v18;
          v22 = sub_B44900(v21);
          v12 = v63;
          v11 = v61;
          v14 = 0x40540000000000LL;
          v16 = v59;
          if ( v22 == (a5 & 4) )
          {
            v23 = sub_B448F0((__int64)(v17 - 3));
            v21 = (__int64)(v17 - 3);
            v12 = v63;
            v14 = 0x40540000000000LL;
            v11 = v61;
            v16 = v59;
            v19 = v57;
            LOBYTE(v18) = v56;
            if ( v23 == (a5 & 2) )
            {
LABEL_21:
              if ( (unsigned int)(v19 - 48) > 1 && (unsigned __int8)(v18 - 55) > 1u )
                return v21;
              v58 = v16;
              v60 = v11;
              v62 = v12;
              v24 = sub_B44E60(v21);
              v12 = v62;
              v14 = 0x40540000000000LL;
              v11 = v60;
              v16 = v58;
              if ( !v24 )
                return v21;
            }
          }
        }
      }
      if ( v13 == v17 )
        break;
      v17 = (_QWORD *)(*v17 & 0xFFFFFFFFFFFFFFF8LL);
      --v15;
    }
    while ( v15 );
    v8 = v16;
  }
  v26 = v12[3];
  v67 = v26;
  if ( v26 )
  {
    sub_B96E90((__int64)&v67, v26, 1);
    v11 = v8[71];
  }
  v73[2] = v11;
  v72 = v8 + 65;
  v73[0] = 0;
  v73[1] = 0;
  if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
    sub_BD73F0((__int64)v73);
  v27 = *((_WORD *)v8 + 292);
  v73[3] = v8[72];
  v74 = v27;
  sub_B33910(v75, v8 + 65);
  v30 = *((unsigned int *)v8 + 198);
  v31 = *((unsigned int *)v8 + 199);
  v75[1] = v8;
  if ( v30 + 1 > v31 )
  {
    sub_C8D5F0((__int64)(v8 + 98), v8 + 100, v30 + 1, 8u, v28, v29);
    v30 = *((unsigned int *)v8 + 198);
  }
  *(_QWORD *)(v8[98] + 8 * v30) = &v72;
  ++*((_DWORD *)v8 + 198);
  if ( a6 )
  {
    while ( 1 )
    {
      v44 = v8[71];
      v45 = *(_QWORD *)(*v8 + 48);
      v46 = *(_QWORD *)(v45 + 8);
      v47 = *(_DWORD *)(v45 + 24);
      if ( !v47 )
        break;
      v48 = v47 - 1;
      v49 = (v47 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v50 = (__int64 *)(v46 + 16 * v49);
      v51 = *v50;
      if ( *v50 != v44 )
      {
        v52 = 1;
        while ( v51 != -4096 )
        {
          v53 = v52 + 1;
          v49 = v48 & (unsigned int)(v52 + v49);
          v50 = (__int64 *)(v46 + 16LL * (unsigned int)v49);
          v51 = *v50;
          if ( *v50 == v44 )
            goto LABEL_49;
          v52 = v53;
        }
        break;
      }
LABEL_49:
      v37 = v50[1];
      if ( !v37 )
        break;
      if ( !(unsigned __int8)sub_D48480(v50[1], (__int64)a3, v49, v44) )
        break;
      if ( !(unsigned __int8)sub_D48480(v37, (__int64)a4, v38, v39) )
        break;
      v40 = sub_D4B130(v37);
      if ( !v40 )
        break;
      v41 = *(_QWORD *)(v40 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v41 == v40 + 48 )
      {
        v43 = 0;
      }
      else
      {
        if ( !v41 )
          BUG();
        v42 = *(unsigned __int8 *)(v41 - 24);
        v43 = v41 - 24;
        if ( (unsigned int)(v42 - 30) >= 0xB )
          v43 = 0;
      }
      sub_D5F1F0((__int64)(v8 + 65), v43);
    }
  }
  v69 = 257;
  v71 = 257;
  v9 = sub_B504D0(a2, (__int64)a3, (__int64)a4, (__int64)v68, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v8[76] + 16LL))(
    v8[76],
    v9,
    v70,
    v8[72],
    v8[73]);
  v32 = v8[65];
  v33 = v32 + 16LL * *((unsigned int *)v8 + 132);
  while ( v33 != v32 )
  {
    v34 = *(_QWORD *)(v32 + 8);
    v35 = *(_DWORD *)v32;
    v32 += 16;
    sub_B99FD0(v9, v35, v34);
  }
  v36 = (__int64 *)(v9 + 48);
  v70[0] = v67;
  if ( !v67 )
  {
    if ( v36 == v70 )
      goto LABEL_45;
    v54 = *(_QWORD *)(v9 + 48);
    if ( !v54 )
      goto LABEL_45;
LABEL_66:
    sub_B91220(v9 + 48, v54);
    goto LABEL_67;
  }
  sub_B96E90((__int64)v70, v67, 1);
  if ( v36 == v70 )
  {
    if ( v70[0] )
      sub_B91220((__int64)v70, v70[0]);
LABEL_45:
    if ( (a5 & 2) == 0 )
      goto LABEL_46;
LABEL_69:
    sub_B447F0((unsigned __int8 *)v9, 1);
    if ( (a5 & 4) == 0 )
      goto LABEL_47;
    goto LABEL_70;
  }
  v54 = *(_QWORD *)(v9 + 48);
  if ( v54 )
    goto LABEL_66;
LABEL_67:
  v55 = (unsigned __int8 *)v70[0];
  *(_QWORD *)(v9 + 48) = v70[0];
  if ( !v55 )
    goto LABEL_45;
  sub_B976B0((__int64)v70, v55, v9 + 48);
  if ( (a5 & 2) != 0 )
    goto LABEL_69;
LABEL_46:
  if ( (a5 & 4) == 0 )
    goto LABEL_47;
LABEL_70:
  sub_B44850((unsigned __int8 *)v9, 1);
LABEL_47:
  sub_F80960((__int64)&v72);
  if ( v67 )
    sub_B91220((__int64)&v67, v67);
  return v9;
}
