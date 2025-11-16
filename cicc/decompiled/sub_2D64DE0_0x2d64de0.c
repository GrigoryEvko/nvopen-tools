// Function: sub_2D64DE0
// Address: 0x2d64de0
//
__int64 __fastcall sub_2D64DE0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r13
  unsigned int v4; // r13d
  unsigned __int8 **v7; // rax
  __int64 v8; // rax
  unsigned int *v9; // rax
  __int64 v10; // rcx
  unsigned int *v11; // rsi
  __int64 v12; // rcx
  signed __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // r10
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 *v26; // rsi
  _BYTE *v27; // r10
  __int64 *v28; // rax
  __int64 v29; // rsi
  int v30; // eax
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // [rsp+0h] [rbp-130h]
  __int64 **v34; // [rsp+8h] [rbp-128h]
  __int64 v35; // [rsp+28h] [rbp-108h]
  _BYTE *v36; // [rsp+28h] [rbp-108h]
  _QWORD *v37; // [rsp+28h] [rbp-108h]
  int v38; // [rsp+38h] [rbp-F8h]
  __int64 v39[2]; // [rsp+40h] [rbp-F0h] BYREF
  __int64 (__fastcall *v40)(__int64 *, __int64 *, int); // [rsp+50h] [rbp-E0h]
  __int64 (__fastcall *v41)(); // [rsp+58h] [rbp-D8h]
  __int16 v42; // [rsp+60h] [rbp-D0h]
  unsigned int *v43; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+78h] [rbp-B8h] BYREF
  _BYTE v45[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+A0h] [rbp-90h]
  __int64 v47; // [rsp+A8h] [rbp-88h]
  __int16 v48; // [rsp+B0h] [rbp-80h]
  __int64 v49; // [rsp+B8h] [rbp-78h]
  void **v50; // [rsp+C0h] [rbp-70h]
  void **v51; // [rsp+C8h] [rbp-68h]
  __int64 v52; // [rsp+D0h] [rbp-60h]
  int v53; // [rsp+D8h] [rbp-58h]
  __int16 v54; // [rsp+DCh] [rbp-54h]
  char v55; // [rsp+DEh] [rbp-52h]
  __int64 v56; // [rsp+E0h] [rbp-50h]
  __int64 v57; // [rsp+E8h] [rbp-48h]
  void *v58; // [rsp+F0h] [rbp-40h] BYREF
  void *v59; // [rsp+F8h] [rbp-38h] BYREF

  v44 = 0;
  if ( !a2 )
    return 0;
  v2 = *(_BYTE **)(a2 - 64);
  if ( *v2 != 91 )
    return 0;
  v7 = (unsigned __int8 **)sub_986520(*(_QWORD *)(a2 - 64));
  if ( !(unsigned __int8)sub_AC2BE0(*v7) )
    return 0;
  v8 = sub_986520((__int64)v2);
  if ( !(unsigned __int8)sub_10081F0((__int64 **)&v44, *(_QWORD *)(v8 + 64)) )
    return 0;
  v4 = sub_AC2BE0(*(unsigned __int8 **)(a2 - 32));
  if ( !(_BYTE)v4 )
    return 0;
  v9 = *(unsigned int **)(a2 + 72);
  v10 = *(unsigned int *)(a2 + 80);
  v11 = &v9[v10];
  v12 = (v10 * 4) >> 4;
  if ( !v12 )
  {
LABEL_33:
    v13 = (char *)v11 - (char *)v9;
    if ( (char *)v11 - (char *)v9 != 8 )
    {
      if ( v13 != 12 )
      {
        if ( v13 != 4 )
          goto LABEL_16;
        goto LABEL_36;
      }
      v12 = *v9;
      v13 = (unsigned int)(v12 + 1);
      if ( (unsigned int)v13 > 1 )
        goto LABEL_15;
      ++v9;
    }
    v12 = *v9;
    v13 = (unsigned int)(v12 + 1);
    if ( (unsigned int)v13 > 1 )
      goto LABEL_15;
    ++v9;
LABEL_36:
    v12 = *v9;
    v13 = (unsigned int)(v12 + 1);
    if ( (unsigned int)v13 <= 1 )
      goto LABEL_16;
    goto LABEL_15;
  }
  v12 = (__int64)&v9[4 * v12];
  while ( 1 )
  {
    v13 = *v9 + 1;
    if ( (unsigned int)v13 > 1 )
      break;
    v13 = v9[1] + 1;
    if ( (unsigned int)v13 > 1 )
    {
      ++v9;
      break;
    }
    v13 = v9[2] + 1;
    if ( (unsigned int)v13 > 1 )
    {
      v9 += 2;
      break;
    }
    v13 = v9[3] + 1;
    if ( (unsigned int)v13 > 1 )
    {
      v9 += 3;
      break;
    }
    v9 += 4;
    if ( (unsigned int *)v12 == v9 )
      goto LABEL_33;
  }
LABEL_15:
  if ( v11 != v9 )
    return 0;
LABEL_16:
  v14 = *(_QWORD *)(a1 + 16);
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 1344LL);
  if ( v15 == sub_2D56650 )
    return 0;
  v16 = (__int64 *)((__int64 (__fastcall *)(__int64, __int64, signed __int64, __int64))v15)(v14, a2, v13, v12);
  if ( !v16 )
    return 0;
  v34 = (__int64 **)v16;
  v33 = *(_QWORD *)(a2 + 8);
  v35 = sub_BCDA70(v16, *(_DWORD *)(v33 + 32));
  v17 = sub_BD5C60(a2);
  v55 = 7;
  v49 = v17;
  v50 = &v58;
  v51 = &v59;
  v43 = (unsigned int *)v45;
  v58 = &unk_49DA100;
  v54 = 512;
  v48 = 0;
  v44 = 0x200000000LL;
  v59 = &unk_49DA0B0;
  v52 = 0;
  v53 = 0;
  v56 = 0;
  v57 = 0;
  v46 = 0;
  v47 = 0;
  sub_D5F1F0((__int64)&v43, a2);
  v18 = *(_QWORD *)(a2 - 64);
  v42 = 257;
  if ( (*(_BYTE *)(v18 + 7) & 0x40) != 0 )
    v19 = *(_QWORD *)(v18 - 8);
  else
    v19 = v18 - 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF);
  v20 = sub_2D5B7B0((__int64 *)&v43, 0x31u, *(_QWORD *)(v19 + 32), v34, (__int64)v39, 0, v38, 0);
  v42 = 257;
  v21 = v35;
  v36 = (_BYTE *)v20;
  v22 = sub_B37A60(&v43, *(_DWORD *)(v21 + 32), v20, v39);
  v42 = 257;
  v23 = sub_2D5B7B0((__int64 *)&v43, 0x31u, v22, (__int64 **)v33, (__int64)v39, 0, v38, 0);
  sub_2D594F0(a2, v23, (__int64 *)(a1 + 840), *(unsigned __int8 *)(a1 + 832), v24, v25);
  v26 = *(__int64 **)(a1 + 48);
  v39[0] = a1;
  v41 = sub_2D69730;
  v40 = sub_2D56BD0;
  sub_F5CAB0((char *)a2, v26, 0, (__int64)v39);
  v27 = v36;
  if ( v40 )
  {
    v40(v39, v39, 3);
    v27 = v36;
  }
  if ( *v27 > 0x1Cu )
  {
    v37 = v27;
    v28 = (__int64 *)sub_986520((__int64)v27);
    v29 = *v28;
    v30 = *(unsigned __int8 *)*v28;
    if ( (unsigned __int8)v30 > 0x1Cu
      && *(_QWORD *)(v29 + 40) != v37[5]
      && (_BYTE)v30 != 84
      && (unsigned int)(v30 - 30) > 0xA )
    {
      v31 = (unsigned int)(v30 - 39);
      if ( (unsigned int)v31 > 0x38 || (v32 = 0x100060000000001LL, !_bittest64(&v32, v31)) )
        sub_B44530(v37, v29);
    }
  }
  nullsub_61();
  v58 = &unk_49DA100;
  nullsub_63();
  if ( v43 != (unsigned int *)v45 )
    _libc_free((unsigned __int64)v43);
  return v4;
}
