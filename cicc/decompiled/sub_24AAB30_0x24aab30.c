// Function: sub_24AAB30
// Address: 0x24aab30
//
void __fastcall sub_24AAB30(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r15
  unsigned int *v10; // rax
  int v11; // ecx
  unsigned int *v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // r12
  __int64 **v15; // r11
  __int64 (__fastcall *v16)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rax
  _QWORD *v28; // rax
  unsigned __int64 v29; // rbx
  unsigned int *v30; // r12
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rax
  unsigned __int64 v34; // rsi
  __int64 **v35; // [rsp+18h] [rbp-148h]
  __int64 **v36; // [rsp+28h] [rbp-138h]
  __int64 v37; // [rsp+28h] [rbp-138h]
  __int64 **v38; // [rsp+28h] [rbp-138h]
  unsigned int v39; // [rsp+38h] [rbp-128h]
  _QWORD v40[4]; // [rsp+40h] [rbp-120h] BYREF
  __int64 v41; // [rsp+60h] [rbp-100h]
  _QWORD v42[4]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v43; // [rsp+90h] [rbp-D0h]
  unsigned int *v44; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+A8h] [rbp-B8h]
  _BYTE v46[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+D0h] [rbp-90h]
  __int64 v48; // [rsp+D8h] [rbp-88h]
  __int64 v49; // [rsp+E0h] [rbp-80h]
  _QWORD *v50; // [rsp+E8h] [rbp-78h]
  void **v51; // [rsp+F0h] [rbp-70h]
  void **v52; // [rsp+F8h] [rbp-68h]
  __int64 v53; // [rsp+100h] [rbp-60h]
  int v54; // [rsp+108h] [rbp-58h]
  __int16 v55; // [rsp+10Ch] [rbp-54h]
  char v56; // [rsp+10Eh] [rbp-52h]
  __int64 v57; // [rsp+110h] [rbp-50h]
  __int64 v58; // [rsp+118h] [rbp-48h]
  void *v59; // [rsp+120h] [rbp-40h] BYREF
  void *v60; // [rsp+128h] [rbp-38h] BYREF

  v3 = a1;
  v35 = *(__int64 ***)(*(_QWORD *)a1 + 40LL);
  v4 = (_QWORD *)sub_BD5C60(a2);
  v56 = 7;
  v50 = v4;
  v51 = &v59;
  v52 = &v60;
  v44 = (unsigned int *)v46;
  v59 = &unk_49DA100;
  v45 = 0x200000000LL;
  v53 = 0;
  v60 = &unk_49DA0B0;
  v5 = *(_QWORD *)(a2 + 40);
  v54 = 0;
  v47 = v5;
  v55 = 512;
  v57 = 0;
  v58 = 0;
  v48 = a2 + 24;
  LOWORD(v49) = 0;
  v6 = *(_QWORD *)sub_B46C60(a2);
  v42[0] = v6;
  if ( v6 && (sub_B96E90((__int64)v42, v6, 1), (v9 = v42[0]) != 0) )
  {
    v10 = v44;
    v11 = v45;
    v12 = &v44[4 * (unsigned int)v45];
    if ( v44 != v12 )
    {
      while ( 1 )
      {
        v8 = *v10;
        if ( !(_DWORD)v8 )
          break;
        v10 += 4;
        if ( v12 == v10 )
          goto LABEL_22;
      }
      *((_QWORD *)v10 + 1) = v42[0];
      goto LABEL_8;
    }
LABEL_22:
    if ( (unsigned int)v45 >= (unsigned __int64)HIDWORD(v45) )
    {
      v34 = (unsigned int)v45 + 1LL;
      if ( HIDWORD(v45) < v34 )
      {
        sub_C8D5F0((__int64)&v44, v46, v34, 0x10u, v7, v8);
        v12 = &v44[4 * (unsigned int)v45];
      }
      *(_QWORD *)v12 = 0;
      *((_QWORD *)v12 + 1) = v9;
      v9 = v42[0];
      LODWORD(v45) = v45 + 1;
    }
    else
    {
      if ( v12 )
      {
        *v12 = 0;
        *((_QWORD *)v12 + 1) = v9;
        v11 = v45;
        v9 = v42[0];
      }
      LODWORD(v45) = v11 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v44, 0);
    v9 = v42[0];
  }
  if ( v9 )
LABEL_8:
    sub_B91220((__int64)v42, v9);
  v13 = sub_BCB2E0(v50);
  v14 = *(_QWORD *)(a2 - 96);
  LOWORD(v41) = 257;
  v15 = (__int64 **)v13;
  if ( v13 == *(_QWORD *)(v14 + 8) )
  {
    v18 = v14;
    goto LABEL_16;
  }
  v16 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v51 + 15);
  if ( v16 != sub_920130 )
  {
    v38 = v15;
    v33 = v16((__int64)v51, 39u, (_BYTE *)v14, (__int64)v15);
    v15 = v38;
    v18 = v33;
    goto LABEL_15;
  }
  if ( *(_BYTE *)v14 <= 0x15u )
  {
    v36 = v15;
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v17 = sub_ADAB70(39, v14, v36, 0);
    else
      v17 = sub_AA93C0(0x27u, v14, (__int64)v36);
    v15 = v36;
    v18 = v17;
LABEL_15:
    if ( v18 )
      goto LABEL_16;
  }
  v37 = (__int64)v15;
  v43 = 257;
  v28 = sub_BD2C40(72, unk_3F10A14);
  v18 = (__int64)v28;
  if ( v28 )
    sub_B515B0((__int64)v28, v14, v37, (__int64)v42, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v52 + 2))(v52, v18, v40, v48, v49);
  if ( v44 != &v44[4 * (unsigned int)v45] )
  {
    v29 = (unsigned __int64)v44;
    v30 = &v44[4 * (unsigned int)v45];
    do
    {
      v31 = *(_QWORD *)(v29 + 8);
      v32 = *(_DWORD *)v29;
      v29 += 16LL;
      sub_B99FD0(v18, v32, v31);
    }
    while ( v30 != (unsigned int *)v29 );
    v3 = a1;
  }
LABEL_16:
  v19 = sub_BCE3C0(*v35, 0);
  v20 = sub_ADB060(*(_QWORD *)(v3 + 32), v19);
  v21 = *(_QWORD *)(v3 + 40);
  v43 = 257;
  v40[0] = v20;
  v22 = sub_BCB2E0(v50);
  v23 = sub_ACD640(v22, v21, 0);
  v24 = *(unsigned int *)(v3 + 24);
  v40[1] = v23;
  v25 = sub_BCB2D0(v50);
  v40[2] = sub_ACD640(v25, v24, 0);
  v26 = **(unsigned int **)(v3 + 16);
  v27 = sub_BCB2D0(v50);
  v40[3] = sub_ACD640(v27, v26, 0);
  v41 = v18;
  sub_B33D10((__int64)&v44, 0xC7u, 0, 0, (int)v40, 5, v39, (__int64)v42);
  ++**(_DWORD **)(v3 + 16);
  nullsub_61();
  v59 = &unk_49DA100;
  nullsub_63();
  if ( v44 != (unsigned int *)v46 )
    _libc_free((unsigned __int64)v44);
}
