// Function: sub_2CD2800
// Address: 0x2cd2800
//
void __fastcall sub_2CD2800(_BYTE *a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 *v5; // rax
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r15
  _QWORD *v8; // r13
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 *v12; // rdi
  unsigned __int64 v13; // rax
  int v14; // edx
  unsigned __int64 v15; // r10
  int v16; // r11d
  __int64 v17; // rax
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  __int64 **v20; // rax
  unsigned __int64 v21; // rax
  __int64 **v22; // rax
  unsigned __int64 v23; // rax
  __int64 **v24; // rax
  __int64 v25; // [rsp+0h] [rbp-1B0h]
  __int64 v26; // [rsp+8h] [rbp-1A8h]
  __int64 v27; // [rsp+10h] [rbp-1A0h]
  __int64 v28; // [rsp+18h] [rbp-198h]
  char v29; // [rsp+18h] [rbp-198h]
  int v31; // [rsp+20h] [rbp-190h]
  unsigned __int64 v33; // [rsp+28h] [rbp-188h]
  char v35; // [rsp+38h] [rbp-178h]
  unsigned __int64 v36; // [rsp+38h] [rbp-178h]
  int v37; // [rsp+48h] [rbp-168h]
  _QWORD v38[2]; // [rsp+50h] [rbp-160h] BYREF
  char v39[32]; // [rsp+60h] [rbp-150h] BYREF
  __int16 v40; // [rsp+80h] [rbp-130h]
  int v41[8]; // [rsp+90h] [rbp-120h] BYREF
  __int16 v42; // [rsp+B0h] [rbp-100h]
  _QWORD v43[2]; // [rsp+C0h] [rbp-F0h] BYREF
  _QWORD v44[2]; // [rsp+D0h] [rbp-E0h] BYREF
  __int16 v45; // [rsp+E0h] [rbp-D0h]
  unsigned int *v46[2]; // [rsp+F0h] [rbp-C0h] BYREF
  char v47; // [rsp+100h] [rbp-B0h] BYREF
  void *v48; // [rsp+170h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(unsigned __int64 **)(a2 - 8);
  else
    v5 = (unsigned __int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = *v5;
  v7 = v5[4];
  v35 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
  if ( *(_BYTE *)(*(_QWORD *)(*v5 + 8) + 8LL) != 5 || *(_BYTE *)(*(_QWORD *)(v7 + 8) + 8LL) != 5 )
  {
    if ( v35 != 5 )
      return;
    v8 = (_QWORD *)sub_BD5C60(a2);
    v27 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL) + 40LL);
    sub_23D0AB0((__int64)v46, a2, 0, 0, 0);
    v29 = 0;
    v10 = *(_QWORD *)(v7 + 8);
    v11 = *(_QWORD *)(v6 + 8);
    goto LABEL_17;
  }
  v8 = (_QWORD *)sub_BD5C60(a2);
  v27 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL) + 40LL);
  sub_23D0AB0((__int64)v46, a2, 0, 0, 0);
  v28 = sub_BCB2F0(v8);
  v9 = sub_BCB2F0(v8);
  v10 = v28;
  v11 = v9;
  if ( v35 == 5 )
  {
    v29 = 1;
LABEL_17:
    v25 = v11;
    v26 = v10;
    v19 = sub_BCB2F0(v8);
    v10 = v26;
    v11 = v25;
    v12 = (__int64 *)v19;
    goto LABEL_10;
  }
  v29 = 1;
  v12 = *(__int64 **)(a2 + 8);
LABEL_10:
  v44[0] = v11;
  v44[1] = v10;
  v43[0] = v44;
  v43[1] = 0x200000002LL;
  v13 = sub_BCF480(v12, v44, 2, 0);
  v15 = sub_BA8C10(v27, a3, a4, v13, 0);
  v16 = v14;
  v45 = 257;
  if ( v29 )
  {
    v31 = v14;
    v33 = v15;
    v40 = 257;
    v20 = (__int64 **)sub_BCB2F0(v8);
    v21 = sub_2CD24F0((__int64 *)v46, 0x31u, v6, v20, (__int64)v39, 0, v41[0], 0);
    v42 = 257;
    v38[0] = v21;
    v22 = (__int64 **)sub_BCB2F0(v8);
    v23 = sub_2CD24F0((__int64 *)v46, 0x31u, v7, v22, (__int64)v41, 0, v37, 0);
    v16 = v31;
    v15 = v33;
    v7 = v23;
  }
  else
  {
    v38[0] = v6;
  }
  v38[1] = v7;
  v17 = sub_921880(v46, v15, v16, (int)v38, 2, (__int64)v43, 0);
  v18 = v17;
  if ( v35 == 5 )
  {
    v36 = v17;
    v45 = 257;
    v24 = (__int64 **)sub_BCB1B0(v8);
    v18 = sub_2CD24F0((__int64 *)v46, 0x31u, v36, v24, (__int64)v43, 0, v41[0], 0);
  }
  sub_BD84D0(a2, v18);
  sub_B43D60((_QWORD *)a2);
  *a1 = 1;
  nullsub_61();
  v48 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v46[0] != &v47 )
    _libc_free((unsigned __int64)v46[0]);
}
