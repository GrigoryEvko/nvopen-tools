// Function: sub_34E2BC0
// Address: 0x34e2bc0
//
__int64 __fastcall sub_34E2BC0(__int64 a1)
{
  unsigned int v2; // ebx
  _QWORD *v3; // rax
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // r8
  unsigned __int64 v11; // rax
  int v12; // ecx
  _QWORD *v13; // rdx
  unsigned __int8 *v14; // r14
  __int64 v16; // rax
  unsigned __int8 *v17; // rbx
  __int64 v18; // rax
  unsigned __int8 *v19; // r10
  __int64 (__fastcall *v20)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v21; // rax
  unsigned __int64 v22; // r13
  _BYTE *v23; // rbx
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rax
  unsigned __int64 v27; // rsi
  unsigned __int8 *v28; // [rsp+8h] [rbp-158h]
  unsigned __int8 *v29; // [rsp+8h] [rbp-158h]
  __int64 v30; // [rsp+8h] [rbp-158h]
  __int64 v31; // [rsp+30h] [rbp-130h] BYREF
  __int64 v32; // [rsp+38h] [rbp-128h]
  const char *v33; // [rsp+40h] [rbp-120h] BYREF
  char v34; // [rsp+60h] [rbp-100h]
  char v35; // [rsp+61h] [rbp-FFh]
  __int64 v36[4]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v37; // [rsp+90h] [rbp-D0h]
  _BYTE *v38; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+A8h] [rbp-B8h]
  _BYTE v40[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v41; // [rsp+D0h] [rbp-90h]
  __int64 v42; // [rsp+D8h] [rbp-88h]
  __int64 v43; // [rsp+E0h] [rbp-80h]
  _QWORD *v44; // [rsp+E8h] [rbp-78h]
  void **v45; // [rsp+F0h] [rbp-70h]
  void **v46; // [rsp+F8h] [rbp-68h]
  __int64 v47; // [rsp+100h] [rbp-60h]
  int v48; // [rsp+108h] [rbp-58h]
  __int16 v49; // [rsp+10Ch] [rbp-54h]
  char v50; // [rsp+10Eh] [rbp-52h]
  __int64 v51; // [rsp+110h] [rbp-50h]
  __int64 v52; // [rsp+118h] [rbp-48h]
  void *v53; // [rsp+120h] [rbp-40h] BYREF
  void *v54; // [rsp+128h] [rbp-38h] BYREF

  v32 = sub_B5A2C0(a1);
  v2 = v32;
  v3 = (_QWORD *)sub_BD5C60(a1);
  v4 = sub_BCB2D0(v3);
  v31 = v4;
  if ( !BYTE4(v32) )
  {
    v14 = (unsigned __int8 *)sub_AD64C0(v4, (unsigned int)v32, 0);
    goto LABEL_12;
  }
  v5 = *(_QWORD *)(a1 + 40);
  v6 = sub_AA48A0(v5);
  v41 = v5;
  v7 = (_QWORD *)v6;
  v46 = &v54;
  v38 = v40;
  v39 = 0x200000000LL;
  v45 = &v53;
  v44 = (_QWORD *)v6;
  v47 = 0;
  v53 = &unk_49DA100;
  v48 = 0;
  v49 = 512;
  v50 = 7;
  v51 = 0;
  v52 = 0;
  v54 = &unk_49DA0B0;
  v42 = a1 + 24;
  LOWORD(v43) = 0;
  if ( a1 + 24 != v5 + 48 )
  {
    v8 = *(_QWORD *)sub_B46C60(a1);
    v36[0] = v8;
    if ( v8 && (sub_B96E90((__int64)v36, v8, 1), (v10 = v36[0]) != 0) )
    {
      v11 = (unsigned __int64)v38;
      v12 = v39;
      v13 = &v38[16 * (unsigned int)v39];
      if ( v38 != (_BYTE *)v13 )
      {
        while ( *(_DWORD *)v11 )
        {
          v11 += 16LL;
          if ( v13 == (_QWORD *)v11 )
            goto LABEL_29;
        }
        *(_QWORD *)(v11 + 8) = v36[0];
LABEL_10:
        sub_B91220((__int64)v36, v10);
LABEL_15:
        v7 = v44;
        goto LABEL_16;
      }
LABEL_29:
      if ( (unsigned int)v39 >= (unsigned __int64)HIDWORD(v39) )
      {
        v27 = (unsigned int)v39 + 1LL;
        if ( HIDWORD(v39) < v27 )
        {
          v30 = v36[0];
          sub_C8D5F0((__int64)&v38, v40, v27, 0x10u, v36[0], v9);
          v10 = v30;
          v13 = &v38[16 * (unsigned int)v39];
        }
        *v13 = 0;
        v13[1] = v10;
        v10 = v36[0];
        LODWORD(v39) = v39 + 1;
      }
      else
      {
        if ( v13 )
        {
          *(_DWORD *)v13 = 0;
          v13[1] = v10;
          v12 = v39;
          v10 = v36[0];
        }
        LODWORD(v39) = v12 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v38, 0);
      v10 = v36[0];
    }
    if ( !v10 )
      goto LABEL_15;
    goto LABEL_10;
  }
LABEL_16:
  v16 = sub_BCB2D0(v7);
  v17 = (unsigned __int8 *)sub_ACD640(v16, v2, 0);
  BYTE4(v33) = 0;
  v36[0] = (__int64)"vscale";
  v37 = 259;
  v18 = sub_B33D10((__int64)&v38, 0x1EDu, (__int64)&v31, 1, 0, 0, (__int64)v33, (__int64)v36);
  v35 = 1;
  v19 = (unsigned __int8 *)v18;
  v34 = 3;
  v33 = "scalable_size";
  v20 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v45 + 4);
  if ( v20 == sub_9201A0 )
  {
    if ( *v19 > 0x15u || *v17 > 0x15u )
      goto LABEL_26;
    v28 = v19;
    if ( (unsigned __int8)sub_AC47B0(17) )
      v21 = sub_AD5570(17, (__int64)v28, v17, 1, 0);
    else
      v21 = sub_AABE40(0x11u, v28, v17);
    v19 = v28;
    v14 = (unsigned __int8 *)v21;
  }
  else
  {
    v29 = v19;
    v26 = v20((__int64)v45, 17u, v19, v17, 1u, 0);
    v19 = v29;
    v14 = (unsigned __int8 *)v26;
  }
  if ( !v14 )
  {
LABEL_26:
    v37 = 257;
    v14 = (unsigned __int8 *)sub_B504D0(17, (__int64)v19, (__int64)v17, (__int64)v36, 0, 0);
    (*((void (__fastcall **)(void **, unsigned __int8 *, const char **, __int64, __int64))*v46 + 2))(
      v46,
      v14,
      &v33,
      v42,
      v43);
    v22 = (unsigned __int64)v38;
    v23 = &v38[16 * (unsigned int)v39];
    if ( v38 != v23 )
    {
      do
      {
        v24 = *(_QWORD *)(v22 + 8);
        v25 = *(_DWORD *)v22;
        v22 += 16LL;
        sub_B99FD0((__int64)v14, v25, v24);
      }
      while ( v23 != (_BYTE *)v22 );
    }
    sub_B447F0(v14, 1);
  }
  nullsub_61();
  v53 = &unk_49DA100;
  nullsub_63();
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
LABEL_12:
  sub_B5A4C0(a1, (__int64)v14);
  return 1;
}
