// Function: sub_2C216C0
// Address: 0x2c216c0
//
void __fastcall sub_2C216C0(__int64 a1, __int64 a2)
{
  __int64 v3; // r10
  __int64 v4; // rax
  unsigned __int64 v5; // rsi
  int v6; // eax
  __int64 v7; // r14
  __int64 v8; // rsi
  char v9; // r14
  __int64 v10; // r10
  __int64 v11; // rax
  _QWORD *v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // r10
  _BYTE *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  _BYTE *v20; // r14
  _BYTE *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-158h]
  __int64 v25; // [rsp+10h] [rbp-150h]
  _BYTE *v26; // [rsp+10h] [rbp-150h]
  __int64 v27; // [rsp+38h] [rbp-128h]
  _BYTE v28[32]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v29; // [rsp+60h] [rbp-100h]
  __int64 v30[4]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v31; // [rsp+90h] [rbp-D0h]
  unsigned int *v32[2]; // [rsp+A0h] [rbp-C0h] BYREF
  _BYTE v33[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+D0h] [rbp-90h]
  __int64 v35; // [rsp+D8h] [rbp-88h]
  __int16 v36; // [rsp+E0h] [rbp-80h]
  __int64 v37; // [rsp+E8h] [rbp-78h]
  void **v38; // [rsp+F0h] [rbp-70h]
  void **v39; // [rsp+F8h] [rbp-68h]
  __int64 v40; // [rsp+100h] [rbp-60h]
  int v41; // [rsp+108h] [rbp-58h]
  __int16 v42; // [rsp+10Ch] [rbp-54h]
  char v43; // [rsp+10Eh] [rbp-52h]
  __int64 v44; // [rsp+110h] [rbp-50h]
  __int64 v45; // [rsp+118h] [rbp-48h]
  void *v46; // [rsp+120h] [rbp-40h] BYREF
  void *v47; // [rsp+128h] [rbp-38h] BYREF

  v3 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 1);
  v24 = *(_QWORD *)(v3 + 8);
  v4 = *(_QWORD *)(a2 + 104);
  v5 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 == v4 + 48 )
  {
    v7 = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    v6 = *(unsigned __int8 *)(v5 - 24);
    v7 = 0;
    v8 = v5 - 24;
    if ( (unsigned int)(v6 - 30) < 0xB )
      v7 = v8;
  }
  v25 = v3;
  v37 = sub_BD5C60(v7);
  v38 = &v46;
  v39 = &v47;
  v32[0] = (unsigned int *)v33;
  v46 = &unk_49DA100;
  v42 = 512;
  v32[1] = (unsigned int *)0x200000000LL;
  v47 = &unk_49DA0B0;
  v40 = 0;
  v41 = 0;
  v43 = 7;
  v44 = 0;
  v45 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  sub_D5F1F0((__int64)v32, v7);
  v9 = *(_BYTE *)(a2 + 12);
  v10 = v25;
  v27 = *(_QWORD *)(a2 + 8);
  if ( v9 != 1 && (unsigned int)*(_QWORD *)(a2 + 8) == 1 )
  {
    if ( *(_DWORD *)(a1 + 56) != 2 || (v13 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL)) == 0 )
    {
      v23 = sub_2AB26E0((__int64)v32, v24, *(_QWORD *)(a2 + 8), 0);
      v16 = (_BYTE *)v25;
      v17 = (_BYTE *)v23;
      goto LABEL_13;
    }
    goto LABEL_9;
  }
  v30[0] = (__int64)"broadcast";
  v31 = 259;
  v11 = sub_B37620(v32, v27, v25, v30);
  LODWORD(v12) = 0;
  v10 = v11;
  if ( *(_DWORD *)(a1 + 56) == 2 )
  {
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
    if ( v13 )
    {
LABEL_9:
      v14 = *(_QWORD *)(v13 + 40);
      v12 = *(_QWORD **)(v14 + 24);
      if ( *(_DWORD *)(v14 + 32) > 0x40u )
        v12 = (_QWORD *)*v12;
    }
  }
  v26 = (_BYTE *)v10;
  v15 = sub_2AB26E0((__int64)v32, v24, v27, (int)v12);
  v16 = v26;
  v17 = (_BYTE *)v15;
  if ( v9 )
  {
    if ( !(_DWORD)v27 )
      goto LABEL_13;
  }
  else if ( (unsigned int)v27 <= 1 )
  {
    goto LABEL_13;
  }
  v31 = 257;
  v19 = sub_B37620(v32, v27, v15, v30);
  v31 = 257;
  v20 = (_BYTE *)v19;
  v29 = 257;
  v21 = (_BYTE *)sub_B33FB0((__int64)v32, *(_QWORD *)(v19 + 8), (__int64)v28);
  v22 = sub_929C50(v32, v20, v21, (__int64)v30, 0, 0);
  v16 = v26;
  v17 = (_BYTE *)v22;
LABEL_13:
  v30[0] = (__int64)"vec.iv";
  v31 = 259;
  v18 = sub_929C50(v32, v16, v17, (__int64)v30, 0, 0);
  sub_2BF26E0(a2, a1 + 96, v18, 0);
  nullsub_61();
  v46 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v32[0] != v33 )
    _libc_free((unsigned __int64)v32[0]);
}
