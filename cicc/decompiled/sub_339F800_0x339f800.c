// Function: sub_339F800
// Address: 0x339f800
//
void __fastcall sub_339F800(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rax
  unsigned int v15; // edx
  unsigned __int16 *v16; // rdx
  __int64 v17; // rax
  unsigned __int16 v18; // r15
  unsigned __int16 v19; // ax
  __int64 v20; // rdi
  _QWORD *v21; // r11
  int v22; // ecx
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rdx
  __int128 v27; // rax
  __int64 v28; // rax
  int v29; // edx
  int v30; // r13d
  __int64 v31; // r12
  _QWORD *v32; // rax
  __int64 v33; // r13
  __int128 v34; // [rsp-30h] [rbp-110h]
  _QWORD *v35; // [rsp+0h] [rbp-E0h]
  int v36; // [rsp+Ch] [rbp-D4h]
  int v37; // [rsp+10h] [rbp-D0h]
  int v38; // [rsp+10h] [rbp-D0h]
  unsigned __int8 v39; // [rsp+18h] [rbp-C8h]
  __int64 v40; // [rsp+18h] [rbp-C8h]
  __int64 v41; // [rsp+20h] [rbp-C0h]
  unsigned __int16 v42; // [rsp+20h] [rbp-C0h]
  __int64 v43; // [rsp+20h] [rbp-C0h]
  __int64 v44; // [rsp+28h] [rbp-B8h]
  __int64 v45; // [rsp+60h] [rbp-80h] BYREF
  int v46; // [rsp+68h] [rbp-78h]
  __int128 v47; // [rsp+70h] [rbp-70h]
  __int64 v48; // [rsp+80h] [rbp-60h]
  __int64 v49[10]; // [rsp+90h] [rbp-50h] BYREF

  v6 = a2;
  v8 = *(_DWORD *)(a1 + 848);
  v9 = *(_QWORD *)a1;
  v45 = 0;
  v46 = v8;
  if ( v9 )
  {
    if ( &v45 != (__int64 *)(v9 + 48) )
    {
      a2 = *(_QWORD *)(v9 + 48);
      v45 = a2;
      if ( a2 )
        sub_B96E90((__int64)&v45, a2, 1);
    }
  }
  v10 = *(unsigned __int16 *)(v6 + 2);
  if ( ((*(_WORD *)(v6 + 2) >> 4) & 0x1Fu) > 0x12 )
    goto LABEL_19;
  LOWORD(v10) = (unsigned __int16)v10 >> 1;
  v36 = dword_44DC240[(*(_WORD *)(v6 + 2) >> 4) & 0x1F];
  v37 = v10 & 7;
  v39 = *(_BYTE *)(v6 + 72);
  v11 = sub_33738B0(a1, a2, v10, (__int64)dword_44DC240, a5, a6);
  v13 = v12;
  v14 = sub_338B750(a1, *(_QWORD *)(v6 - 32));
  v16 = (unsigned __int16 *)(*(_QWORD *)(v14 + 48) + 16LL * v15);
  v17 = *(_QWORD *)(a1 + 864);
  v18 = *v16;
  v41 = *(_QWORD *)(v17 + 16);
  sub_2E79000(*(__int64 **)(v17 + 40));
  v19 = sub_2FEC630(v41, (_BYTE *)v6);
  v20 = *(_QWORD *)(a1 + 864);
  v42 = v19;
  v21 = *(_QWORD **)(v20 + 40);
  memset(v49, 0, 32);
  v35 = v21;
  v22 = sub_33CC4A0(v20, v18, 0);
  if ( v18 <= 1u || (unsigned __int16)(v18 - 504) <= 7u )
LABEL_19:
    BUG();
  v23 = (unsigned __int64)(*(_QWORD *)&byte_444C4A0[16 * v18 - 16] + 7LL) >> 3;
  v24 = *(_QWORD *)(v6 - 64);
  if ( v24 )
  {
    *((_QWORD *)&v47 + 1) = 0;
    BYTE4(v48) = 0;
    *(_QWORD *)&v47 = v24 & 0xFFFFFFFFFFFFFFFBLL;
    v25 = *(_QWORD *)(v24 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17 <= 1 )
      v25 = **(_QWORD **)(v25 + 16);
    LODWORD(v24) = *(_DWORD *)(v25 + 8) >> 8;
  }
  else
  {
    v47 = 0u;
    BYTE4(v48) = 0;
  }
  LODWORD(v48) = v24;
  v38 = sub_2E7BD70(v35, v42, v23, v22, (int)v49, 0, v47, v48, v39, v37, 0);
  v40 = *(_QWORD *)(a1 + 864);
  v43 = sub_338B750(a1, *(_QWORD *)(v6 - 32));
  v44 = v26;
  *(_QWORD *)&v27 = sub_338B750(a1, *(_QWORD *)(v6 - 64));
  *((_QWORD *)&v34 + 1) = v13;
  *(_QWORD *)&v34 = v11;
  v28 = sub_33F34C0(v40, v36, (unsigned int)&v45, v18, 0, v38, v34, v27, v43, v44);
  v30 = v29;
  v31 = v28;
  v49[0] = v6;
  v32 = sub_337DC20(a1 + 8, v49);
  *v32 = v31;
  *((_DWORD *)v32 + 2) = v30;
  v33 = *(_QWORD *)(a1 + 864);
  if ( v31 )
  {
    nullsub_1875(v31, v33, 0);
    *(_QWORD *)(v33 + 384) = v31;
    *(_DWORD *)(v33 + 392) = 1;
    sub_33E2B60(v33, 0);
  }
  else
  {
    *(_QWORD *)(v33 + 384) = 0;
    *(_DWORD *)(v33 + 392) = 1;
  }
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
}
