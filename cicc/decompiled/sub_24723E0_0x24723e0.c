// Function: sub_24723E0
// Address: 0x24723e0
//
void __fastcall sub_24723E0(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // rdx
  __int64 v4; // r15
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 **v8; // rcx
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int **v14; // rsi
  unsigned __int64 v15; // rdi
  unsigned __int8 *v16; // r15
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r9
  unsigned int **v22; // [rsp-8h] [rbp-158h]
  unsigned int v23; // [rsp+4h] [rbp-14Ch]
  __int64 v24; // [rsp+10h] [rbp-140h]
  __int64 v25; // [rsp+20h] [rbp-130h]
  __int64 v26; // [rsp+20h] [rbp-130h]
  __int64 v27; // [rsp+28h] [rbp-128h]
  char v28; // [rsp+37h] [rbp-119h] BYREF
  __int64 v29; // [rsp+38h] [rbp-118h]
  _QWORD *v30; // [rsp+40h] [rbp-110h] BYREF
  __int64 v31; // [rsp+48h] [rbp-108h]
  _QWORD v32[2]; // [rsp+50h] [rbp-100h] BYREF
  unsigned __int64 v33; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v34; // [rsp+68h] [rbp-E8h]
  __int16 v35; // [rsp+80h] [rbp-D0h]
  unsigned int *v36[24]; // [rsp+90h] [rbp-C0h] BYREF

  sub_23D0AB0((__int64)v36, a2, 0, 0, 0);
  v2 = *(_DWORD *)(a2 + 4);
  v28 = 0;
  v3 = v2 & 0x7FFFFFF;
  v4 = *(_QWORD *)(a2 - 32 * v3);
  v25 = *(_QWORD *)(a2 + 32 * (1 - v3));
  v27 = *(_QWORD *)(a2 + 32 * (2 - v3));
  v5 = sub_246F3F0(a1, v27);
  if ( (_BYTE)qword_4FE84C8 )
  {
    sub_2472230(a1, v4, a2);
    sub_2472230(a1, v25, a2);
  }
  if ( **(_BYTE **)(a1 + 8) )
    v6 = (unsigned __int64)sub_2465B30((__int64 *)a1, v4, (__int64)v36, *(_QWORD *)(v5 + 8), 1);
  else
    v6 = sub_2463FC0(a1, v4, v36, 0x100u);
  v24 = v7;
  v32[0] = v6;
  v35 = 257;
  v32[1] = v25;
  v31 = 0x200000002LL;
  v8 = *(__int64 ***)(v27 + 8);
  v30 = v32;
  v9 = sub_24633A0((__int64 *)v36, 0x31u, v5, v8, (__int64)&v33, 0, v29, 0);
  sub_C8D5F0((__int64)&v30, v32, 3u, 8u, (__int64)&v33, v21);
  HIDWORD(v29) = 0;
  v30[(unsigned int)v31] = v9;
  v35 = 257;
  v10 = (unsigned int)(v31 + 1);
  v11 = *(_QWORD *)(a2 - 32);
  LODWORD(v31) = v31 + 1;
  v26 = (__int64)v30;
  if ( !v11 || *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v23 = *(_DWORD *)(v11 + 36);
  v12 = sub_BCB120((_QWORD *)v36[9]);
  v13 = sub_B35180((__int64)v36, v12, v23, v26, v10, v29, (__int64)&v33);
  sub_246EF60(a1, a2, v13);
  v14 = v22;
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
  {
    v16 = (unsigned __int8 *)&v28;
    v17 = sub_B2BEC0(*(_QWORD *)a1);
    if ( byte_4FE8EA9 )
      v16 = (unsigned __int8 *)&byte_4FE8EA9;
    v18 = sub_9208B0(v17, *(_QWORD *)(v5 + 8));
    v34 = v19;
    v33 = (unsigned __int64)(v18 + 7) >> 3;
    v20 = sub_246EE10(a1, v27);
    v14 = v36;
    sub_24677C0((__int64 *)a1, (__int64)v36, v20, v24, v33, v34, *v16);
    v15 = (unsigned __int64)v30;
    if ( v30 != v32 )
      goto LABEL_10;
  }
  else
  {
    v15 = (unsigned __int64)v30;
    if ( v30 != v32 )
LABEL_10:
      _libc_free(v15);
  }
  sub_F94A20(v36, (__int64)v14);
}
