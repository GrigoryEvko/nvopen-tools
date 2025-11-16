// Function: sub_17779C0
// Address: 0x17779c0
//
_QWORD *__fastcall sub_17779C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v7; // rax
  unsigned int v8; // r15d
  int v9; // r15d
  bool v10; // zf
  int v11; // eax
  __int64 v12; // r13
  __int64 **v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r13
  char v17; // al
  __int64 v18; // rdx
  _QWORD *v19; // r14
  __int64 v20; // rdi
  unsigned __int64 v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rsi
  _QWORD *v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  __int16 v29; // dx
  int v30; // eax
  _BYTE *v31; // r13
  _BYTE *v32; // r15
  int v34; // eax
  __int64 *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // r15
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rsi
  __int64 v43; // rsi
  unsigned __int8 *v44; // rsi
  unsigned __int8 v45; // [rsp+8h] [rbp-168h]
  unsigned __int64 *v46; // [rsp+8h] [rbp-168h]
  __int64 v47; // [rsp+18h] [rbp-158h]
  unsigned int v48; // [rsp+18h] [rbp-158h]
  __int64 v51; // [rsp+30h] [rbp-140h] BYREF
  _QWORD *v52; // [rsp+38h] [rbp-138h] BYREF
  _QWORD v53[2]; // [rsp+40h] [rbp-130h] BYREF
  __int64 v54[2]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v55; // [rsp+60h] [rbp-110h]
  __int64 v56[2]; // [rsp+70h] [rbp-100h] BYREF
  __int16 v57; // [rsp+80h] [rbp-F0h]
  _QWORD v58[2]; // [rsp+90h] [rbp-E0h] BYREF
  __int16 v59; // [rsp+A0h] [rbp-D0h]
  _BYTE *v60; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v61; // [rsp+B8h] [rbp-B8h]
  _BYTE v62[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a2 - 24);
  v7 = *(_QWORD *)v6;
  if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) == 16 )
    v7 = **(_QWORD **)(v7 + 16);
  v8 = *(_DWORD *)(v7 + 8);
  v60 = v62;
  v9 = v8 >> 8;
  v10 = *(_QWORD *)(a2 + 48) == 0;
  v61 = 0x800000000LL;
  if ( !v10 || *(__int16 *)(a2 + 18) < 0 )
    sub_161F840(a2, (__int64)&v60);
  v11 = *(unsigned __int8 *)(v6 + 16);
  if ( (unsigned __int8)v11 > 0x17u )
  {
    v34 = v11 - 24;
  }
  else
  {
    if ( (_BYTE)v11 != 5 )
      goto LABEL_7;
    v34 = *(unsigned __int16 *)(v6 + 18);
  }
  if ( v34 == 47 )
  {
    v35 = (*(_BYTE *)(v6 + 23) & 0x40) != 0
        ? *(__int64 **)(v6 - 8)
        : (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    v36 = *v35;
    if ( *v35 )
    {
      v37 = *(_QWORD *)v36;
      if ( **(_QWORD **)(*(_QWORD *)v36 + 16LL) == a3 )
      {
        if ( *(_BYTE *)(v37 + 8) == 16 )
          v37 = a3;
        if ( *(_DWORD *)(v37 + 8) >> 8 == v9 )
        {
          v6 = v36;
          goto LABEL_12;
        }
      }
    }
  }
LABEL_7:
  v55 = 257;
  v12 = *(_QWORD *)(a1 + 8);
  v13 = (__int64 **)sub_1647190((__int64 *)a3, v9);
  if ( v13 != *(__int64 ***)v6 )
  {
    if ( *(_BYTE *)(v6 + 16) > 0x10u )
    {
      v59 = 257;
      v6 = sub_15FDBD0(47, v6, (__int64)v13, (__int64)v58, 0);
      v38 = *(_QWORD *)(v12 + 8);
      if ( v38 )
      {
        v39 = *(__int64 **)(v12 + 16);
        sub_157E9D0(v38 + 40, v6);
        v40 = *(_QWORD *)(v6 + 24);
        v41 = *v39;
        *(_QWORD *)(v6 + 32) = v39;
        v41 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v6 + 24) = v41 | v40 & 7;
        *(_QWORD *)(v41 + 8) = v6 + 24;
        *v39 = *v39 & 7 | (v6 + 24);
      }
      v23 = v54;
      v24 = (_QWORD *)v6;
      sub_164B780(v6, v54);
      v10 = *(_QWORD *)(v12 + 80) == 0;
      v51 = v6;
      if ( v10 )
LABEL_64:
        sub_4263D6(v24, v23, v25);
      (*(void (__fastcall **)(__int64, __int64 *))(v12 + 88))(v12 + 64, &v51);
      v42 = *(_QWORD *)v12;
      if ( *(_QWORD *)v12 )
      {
        v56[0] = *(_QWORD *)v12;
        sub_1623A60((__int64)v56, v42, 2);
        v43 = *(_QWORD *)(v6 + 48);
        if ( v43 )
          sub_161E7C0(v6 + 48, v43);
        v44 = (unsigned __int8 *)v56[0];
        *(_QWORD *)(v6 + 48) = v56[0];
        if ( v44 )
          sub_1623210((__int64)v56, v44, v6 + 48);
      }
    }
    else
    {
      v47 = sub_15A46C0(47, (__int64 ***)v6, v13, 0);
      v14 = sub_14DBA30(v47, *(_QWORD *)(v12 + 96), 0);
      v15 = v47;
      if ( v14 )
        v15 = v14;
      v6 = v15;
    }
  }
LABEL_12:
  v16 = *(_QWORD *)(a1 + 8);
  v53[0] = sub_1649960(a2);
  v17 = *(_BYTE *)(a4 + 16);
  v53[1] = v18;
  if ( v17 )
  {
    if ( v17 == 1 )
    {
      v56[0] = (__int64)v53;
      v57 = 261;
    }
    else
    {
      if ( *(_BYTE *)(a4 + 17) == 1 )
        a4 = *(_QWORD *)a4;
      else
        v17 = 2;
      v56[1] = a4;
      v56[0] = (__int64)v53;
      LOBYTE(v57) = 5;
      HIBYTE(v57) = v17;
    }
  }
  else
  {
    v57 = 256;
  }
  v45 = *(_WORD *)(a2 + 18) & 1;
  v48 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
  v19 = sub_1648A60(64, 1u);
  if ( v19 )
    sub_15F9210((__int64)v19, *(_QWORD *)(*(_QWORD *)v6 + 24LL), v6, 0, v45, 0);
  v20 = *(_QWORD *)(v16 + 8);
  if ( v20 )
  {
    v46 = *(unsigned __int64 **)(v16 + 16);
    sub_157E9D0(v20 + 40, (__int64)v19);
    v21 = *v46;
    v22 = v19[3] & 7LL;
    v19[4] = v46;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    v19[3] = v21 | v22;
    *(_QWORD *)(v21 + 8) = v19 + 3;
    *v46 = *v46 & 7 | (unsigned __int64)(v19 + 3);
  }
  v23 = v56;
  v24 = v19;
  sub_164B780((__int64)v19, v56);
  v10 = *(_QWORD *)(v16 + 80) == 0;
  v52 = v19;
  if ( v10 )
    goto LABEL_64;
  (*(void (__fastcall **)(__int64, _QWORD **))(v16 + 88))(v16 + 64, &v52);
  v26 = *(_QWORD *)v16;
  if ( *(_QWORD *)v16 )
  {
    v58[0] = *(_QWORD *)v16;
    sub_1623A60((__int64)v58, v26, 2);
    v27 = v19[6];
    if ( v27 )
      sub_161E7C0((__int64)(v19 + 6), v27);
    v28 = (unsigned __int8 *)v58[0];
    v19[6] = v58[0];
    if ( v28 )
      sub_1623210((__int64)v58, v28, (__int64)(v19 + 6));
  }
  sub_15F8F50((__int64)v19, v48);
  v29 = *((_WORD *)v19 + 9);
  v30 = *(unsigned __int16 *)(a2 + 18) >> 7;
  *((_BYTE *)v19 + 56) = *(_BYTE *)(a2 + 56);
  *((_WORD *)v19 + 9) = v29 & 0x8000 | v29 & 0x7C7F | ((v30 & 7) << 7);
  sub_16498A0((__int64)v19);
  v31 = v60;
  v32 = &v60[16 * (unsigned int)v61];
  if ( v60 != v32 )
  {
    do
    {
      switch ( *(_DWORD *)v31 )
      {
        case 0:
        case 1:
        case 2:
        case 3:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 0xA:
          goto LABEL_27;
        case 4:
          sub_1AECA20(*(_QWORD *)(a1 + 2664), a2, *((_QWORD *)v31 + 1), v19);
          break;
        case 0xB:
          sub_1AEC950(a2, *((_QWORD *)v31 + 1), v19);
          break;
        case 0xC:
        case 0xD:
        case 0x11:
          if ( *(_BYTE *)(a3 + 8) == 15 )
LABEL_27:
            sub_1625C10((__int64)v19, *(_DWORD *)v31, *((_QWORD *)v31 + 1));
          break;
        default:
          break;
      }
      v31 += 16;
    }
    while ( v32 != v31 );
    v32 = v60;
  }
  if ( v32 != v62 )
    _libc_free((unsigned __int64)v32);
  return v19;
}
