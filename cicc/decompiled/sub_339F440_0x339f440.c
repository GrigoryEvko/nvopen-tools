// Function: sub_339F440
// Address: 0x339f440
//
void __fastcall sub_339F440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  int v8; // edx
  __int64 v9; // rax
  unsigned __int8 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned __int16 v14; // r13
  int v15; // r9d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  unsigned __int16 v19; // ax
  __int64 v20; // rdi
  _QWORD *v21; // r14
  int v22; // ecx
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // r14
  __int64 v27; // rdx
  __int64 v28; // r15
  __int64 v29; // rdx
  __int128 v30; // rax
  __int64 v31; // r14
  int v32; // edx
  _QWORD *v33; // rax
  __int64 v34; // r12
  __int128 v35; // [rsp-10h] [rbp-110h]
  __int64 v36; // [rsp+10h] [rbp-F0h]
  __int64 v37; // [rsp+18h] [rbp-E8h]
  __int128 v38; // [rsp+20h] [rbp-E0h]
  int v39; // [rsp+30h] [rbp-D0h]
  int v40; // [rsp+30h] [rbp-D0h]
  int v41; // [rsp+38h] [rbp-C8h]
  __int64 v42; // [rsp+38h] [rbp-C8h]
  unsigned __int16 v43; // [rsp+40h] [rbp-C0h]
  __int128 v44; // [rsp+40h] [rbp-C0h]
  __int64 v45; // [rsp+80h] [rbp-80h] BYREF
  int v46; // [rsp+88h] [rbp-78h]
  __int128 v47; // [rsp+90h] [rbp-70h]
  __int64 v48; // [rsp+A0h] [rbp-60h]
  __int64 v49[10]; // [rsp+B0h] [rbp-50h] BYREF

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
  v10 = *(_BYTE *)(v6 + 72);
  v41 = (*(_WORD *)(v6 + 2) >> 5) & 7;
  v39 = (*(_WORD *)(v6 + 2) >> 2) & 7;
  *(_QWORD *)&v38 = sub_33738B0(a1, a2, *(_WORD *)(v6 + 2) >> 2, (*(_WORD *)(v6 + 2) >> 2) & 7, a5, a6);
  *((_QWORD *)&v38 + 1) = v11;
  v12 = sub_338B750(a1, *(_QWORD *)(v6 - 64));
  v14 = *(_WORD *)(*(_QWORD *)(v12 + 48) + 16LL * v13);
  v36 = sub_33E5B50(*(_QWORD *)(a1 + 864), v14, 0, 2, 0, v15, 1, 0);
  v16 = *(_QWORD *)(a1 + 864);
  v37 = v17;
  v18 = *(_QWORD *)(v16 + 16);
  sub_2E79000(*(__int64 **)(v16 + 40));
  v19 = sub_2FEC630(v18, (_BYTE *)v6);
  v20 = *(_QWORD *)(a1 + 864);
  v43 = v19;
  v21 = *(_QWORD **)(v20 + 40);
  memset(v49, 0, 32);
  v22 = sub_33CC4A0(v20, v14, 0);
  if ( v14 <= 1u || (unsigned __int16)(v14 - 504) <= 7u )
    BUG();
  v23 = (unsigned __int64)(*(_QWORD *)&byte_444C4A0[16 * v14 - 16] + 7LL) >> 3;
  v24 = *(_QWORD *)(v6 - 96);
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
  v40 = sub_2E7BD70(v21, v43, v23, v22, (int)v49, 0, v47, v48, v10, v39, v41);
  v42 = *(_QWORD *)(a1 + 864);
  v26 = sub_338B750(a1, *(_QWORD *)(v6 - 32));
  v28 = v27;
  *(_QWORD *)&v44 = sub_338B750(a1, *(_QWORD *)(v6 - 64));
  *((_QWORD *)&v44 + 1) = v29;
  *(_QWORD *)&v30 = sub_338B750(a1, *(_QWORD *)(v6 - 96));
  *((_QWORD *)&v35 + 1) = v28;
  *(_QWORD *)&v35 = v26;
  v31 = sub_33E6F00(v42, 341, (unsigned int)&v45, v14, 0, v40, v36, v37, v38, v30, v44, v35);
  LODWORD(v28) = v32;
  v49[0] = v6;
  v33 = sub_337DC20(a1 + 8, v49);
  *v33 = v31;
  *((_DWORD *)v33 + 2) = v28;
  v34 = *(_QWORD *)(a1 + 864);
  if ( v31 )
  {
    nullsub_1875(v31, v34, 0);
    *(_QWORD *)(v34 + 384) = v31;
    *(_DWORD *)(v34 + 392) = 2;
    sub_33E2B60(v34, 0);
  }
  else
  {
    *(_QWORD *)(v34 + 384) = 0;
    *(_DWORD *)(v34 + 392) = 2;
  }
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
}
