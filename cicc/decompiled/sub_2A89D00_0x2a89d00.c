// Function: sub_2A89D00
// Address: 0x2a89d00
//
_BYTE *__fastcall sub_2A89D00(unsigned __int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v8; // r12
  _BYTE *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r9
  __int64 v13; // r8
  unsigned int *v14; // rax
  int v15; // ecx
  unsigned int *v16; // rdx
  __int64 v17; // rsi
  char v18; // al
  char v19; // dl
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int64 v23; // r10
  __int64 v24; // rax
  char v25; // cl
  __int64 **v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // rsi
  __int64 **v29; // rbx
  unsigned int v30; // esi
  _BYTE *v31; // r12
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 **v35; // rax
  unsigned __int64 v36; // rsi
  unsigned __int64 v37; // [rsp+8h] [rbp-138h]
  unsigned __int64 v38; // [rsp+8h] [rbp-138h]
  _QWORD *v39; // [rsp+10h] [rbp-130h]
  unsigned __int64 v41; // [rsp+18h] [rbp-128h]
  unsigned __int64 v42; // [rsp+38h] [rbp-108h]
  int v43; // [rsp+38h] [rbp-108h]
  __int64 v44; // [rsp+38h] [rbp-108h]
  int v45; // [rsp+48h] [rbp-F8h]
  __int64 v46; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v47; // [rsp+58h] [rbp-E8h]
  __int16 v48; // [rsp+70h] [rbp-D0h]
  unsigned int *v49; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+88h] [rbp-B8h]
  _BYTE v51[32]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+B0h] [rbp-90h]
  __int64 v53; // [rsp+B8h] [rbp-88h]
  __int16 v54; // [rsp+C0h] [rbp-80h]
  __int64 v55; // [rsp+C8h] [rbp-78h]
  void **v56; // [rsp+D0h] [rbp-70h]
  void **v57; // [rsp+D8h] [rbp-68h]
  __int64 v58; // [rsp+E0h] [rbp-60h]
  int v59; // [rsp+E8h] [rbp-58h]
  __int16 v60; // [rsp+ECh] [rbp-54h]
  char v61; // [rsp+EEh] [rbp-52h]
  __int64 v62; // [rsp+F0h] [rbp-50h]
  __int64 v63; // [rsp+F8h] [rbp-48h]
  void *v64; // [rsp+100h] [rbp-40h] BYREF
  void *v65; // [rsp+108h] [rbp-38h] BYREF

  v8 = a1;
  v9 = (_BYTE *)sub_B2BEC0(a5);
  v61 = 7;
  v55 = sub_BD5C60(a4);
  v56 = &v64;
  v57 = &v65;
  v49 = (unsigned int *)v51;
  v64 = &unk_49DA100;
  v50 = 0x200000000LL;
  v58 = 0;
  v65 = &unk_49DA0B0;
  v10 = *(_QWORD *)(a4 + 40);
  v59 = 0;
  v52 = v10;
  v60 = 512;
  v62 = 0;
  v63 = 0;
  v53 = a4 + 24;
  v54 = 0;
  v11 = *(_QWORD *)sub_B46C60(a4);
  v46 = v11;
  if ( v11 && (sub_B96E90((__int64)&v46, v11, 1), (v13 = v46) != 0) )
  {
    v14 = v49;
    v15 = v50;
    v16 = &v49[4 * (unsigned int)v50];
    if ( v49 != v16 )
    {
      while ( 1 )
      {
        v12 = *v14;
        if ( !(_DWORD)v12 )
          break;
        v14 += 4;
        if ( v16 == v14 )
          goto LABEL_31;
      }
      *((_QWORD *)v14 + 1) = v46;
      goto LABEL_8;
    }
LABEL_31:
    if ( (unsigned int)v50 >= (unsigned __int64)HIDWORD(v50) )
    {
      v36 = (unsigned int)v50 + 1LL;
      if ( HIDWORD(v50) < v36 )
      {
        v44 = v46;
        sub_C8D5F0((__int64)&v49, v51, v36, 0x10u, v46, v12);
        v13 = v44;
        v16 = &v49[4 * (unsigned int)v50];
      }
      *(_QWORD *)v16 = 0;
      *((_QWORD *)v16 + 1) = v13;
      v13 = v46;
      LODWORD(v50) = v50 + 1;
    }
    else
    {
      if ( v16 )
      {
        *v16 = 0;
        *((_QWORD *)v16 + 1) = v13;
        v15 = v50;
        v13 = v46;
      }
      LODWORD(v50) = v15 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v49, 0);
    v13 = v46;
  }
  if ( v13 )
LABEL_8:
    sub_B91220((__int64)&v46, v13);
  v17 = *(_QWORD *)(a1 + 8);
  v18 = *(_BYTE *)(a3 + 8);
  v19 = *(_BYTE *)(v17 + 8);
  if ( v19 == 14 )
  {
    if ( v18 == 14 )
    {
      if ( *(_DWORD *)(v17 + 8) >> 8 == *(_DWORD *)(a3 + 8) >> 8 )
        goto LABEL_25;
    }
    else if ( v18 == 18 )
    {
      goto LABEL_25;
    }
  }
  else if ( v18 == 18 || v19 == 18 && v18 == 17 )
  {
    goto LABEL_25;
  }
  v39 = *(_QWORD **)v17;
  v46 = sub_9208B0((__int64)v9, v17);
  v47 = v20;
  v42 = (unsigned __int64)(v46 + 7) >> 3;
  v24 = sub_9208B0((__int64)v9, a3);
  v21 = *(_QWORD *)(a1 + 8);
  v46 = v24;
  v47 = v22;
  v23 = (unsigned __int64)(v24 + 7) >> 3;
  LODWORD(v24) = *(unsigned __int8 *)(v21 + 8);
  v25 = *(_BYTE *)(v21 + 8);
  if ( (unsigned int)(v24 - 17) <= 1 )
    v25 = *(_BYTE *)(**(_QWORD **)(v21 + 16) + 8LL);
  if ( v25 == 14 )
  {
    v38 = v23;
    v48 = 257;
    v35 = (__int64 **)sub_AE4450((__int64)v9, v21);
    v24 = sub_2A882B0((__int64 *)&v49, 0x2Fu, a1, v35, (__int64)&v46, 0, v45, 0);
    v23 = v38;
    v8 = v24;
    LOBYTE(v24) = *(_BYTE *)(*(_QWORD *)(v24 + 8) + 8LL);
  }
  if ( (_BYTE)v24 != 12 )
  {
    v37 = v23;
    v48 = 257;
    v26 = (__int64 **)sub_BCCE00(v39, 8 * (int)v42);
    v27 = sub_2A882B0((__int64 *)&v49, 0x31u, v8, v26, (__int64)&v46, 0, v45, 0);
    v23 = v37;
    v8 = v27;
  }
  if ( *v9 )
  {
    v28 = (unsigned int)(8 * (v42 - v23 - a2));
    if ( !(_DWORD)v28 )
      goto LABEL_21;
  }
  else
  {
    v28 = (unsigned int)(8 * a2);
    if ( !(_DWORD)v28 )
      goto LABEL_21;
  }
  v41 = v23;
  v48 = 257;
  v33 = sub_AD64C0(*(_QWORD *)(v8 + 8), v28, 0);
  v34 = sub_F94560((__int64 *)&v49, v8, v33, (__int64)&v46, 0);
  v23 = v41;
  v8 = v34;
LABEL_21:
  if ( v42 != v23 )
  {
    v48 = 257;
    v29 = (__int64 **)sub_BCCE00(v39, 8 * (int)v23);
    v43 = sub_BCB060(*(_QWORD *)(v8 + 8));
    v30 = 49;
    if ( v43 != (unsigned int)sub_BCB060((__int64)v29) )
      v30 = 38;
    v8 = sub_2A882B0((__int64 *)&v49, v30, v8, v29, (__int64)&v46, 0, v45, 0);
  }
LABEL_25:
  v31 = sub_2A88A40((_BYTE *)v8, a3, (__int64 *)&v49, a5);
  nullsub_61();
  v64 = &unk_49DA100;
  nullsub_63();
  if ( v49 != (unsigned int *)v51 )
    _libc_free((unsigned __int64)v49);
  return v31;
}
