// Function: sub_2AD5B30
// Address: 0x2ad5b30
//
_QWORD *__fastcall sub_2AD5B30(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v9; // rdx
  unsigned __int64 v10; // r12
  unsigned int v11; // esi
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r15
  unsigned int v15; // edi
  __int64 *v16; // rdx
  __int64 v17; // r9
  _QWORD *v18; // rdx
  __int64 v19; // r15
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // r12
  unsigned int v23; // esi
  __int64 *v24; // rdx
  __int64 v25; // r11
  __int64 v26; // r9
  unsigned __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 v31; // rdx
  unsigned __int64 v32; // r8
  __int64 v33; // r8
  int v34; // edx
  int v35; // r10d
  __int64 *v36; // r8
  int v37; // eax
  __int64 *v38; // r15
  __int64 v39; // rcx
  _QWORD *v40; // r12
  __int64 v41; // rcx
  int v43; // [rsp+8h] [rbp-98h]
  __int64 v44; // [rsp+8h] [rbp-98h]
  unsigned int v45; // [rsp+10h] [rbp-90h]
  __int64 v46; // [rsp+10h] [rbp-90h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+18h] [rbp-88h]
  __int64 v49; // [rsp+18h] [rbp-88h]
  __int64 *v50; // [rsp+28h] [rbp-78h] BYREF
  __int64 v51; // [rsp+30h] [rbp-70h] BYREF
  __int64 v52; // [rsp+38h] [rbp-68h]
  __int64 v53; // [rsp+40h] [rbp-60h]
  unsigned int v54; // [rsp+48h] [rbp-58h]
  __int64 *v55; // [rsp+50h] [rbp-50h] BYREF
  __int64 v56; // [rsp+58h] [rbp-48h] BYREF
  _BYTE v57[64]; // [rsp+60h] [rbp-40h] BYREF

  v5 = *(_DWORD *)(a2 + 4);
  v53 = 0;
  v54 = 0;
  v51 = 0;
  v45 = v5 & 0x7FFFFFF;
  v6 = *(_QWORD *)(a2 + 40);
  v52 = 0;
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 )
    goto LABEL_11;
  while ( 1 )
  {
    v9 = *(_QWORD *)(v7 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
      break;
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      goto LABEL_11;
  }
  v10 = 0;
  v11 = 0;
  v12 = 0;
LABEL_6:
  v13 = *(_QWORD *)(v9 + 40);
  v14 = *(_QWORD *)(a3 + 8 * v10);
  v55 = (__int64 *)v10;
  v56 = v13;
  if ( !v11 )
  {
    ++v51;
    v50 = 0;
LABEL_48:
    v44 = a3;
    v11 *= 2;
LABEL_49:
    sub_2AD5550((__int64)&v51, v11);
    sub_2AC1AC0((__int64)&v51, &v56, &v50);
    v13 = v56;
    v36 = v50;
    a3 = v44;
    v37 = v53 + 1;
    goto LABEL_35;
  }
  v15 = (v11 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v16 = (__int64 *)(v12 + 16LL * v15);
  v17 = *v16;
  if ( v13 == *v16 )
  {
LABEL_8:
    v18 = v16 + 1;
    goto LABEL_9;
  }
  v43 = 1;
  v36 = 0;
  while ( v17 != -4096 )
  {
    if ( v17 == -8192 && !v36 )
      v36 = v16;
    v15 = (v11 - 1) & (v43 + v15);
    v16 = (__int64 *)(v12 + 16LL * v15);
    v17 = *v16;
    if ( v13 == *v16 )
      goto LABEL_8;
    ++v43;
  }
  if ( !v36 )
    v36 = v16;
  ++v51;
  v37 = v53 + 1;
  v50 = v36;
  if ( 4 * ((int)v53 + 1) >= 3 * v11 )
    goto LABEL_48;
  if ( v11 - (v37 + HIDWORD(v53)) <= v11 >> 3 )
  {
    v44 = a3;
    goto LABEL_49;
  }
LABEL_35:
  LODWORD(v53) = v37;
  if ( *v36 != -4096 )
    --HIDWORD(v53);
  *v36 = v13;
  v18 = v36 + 1;
  v36[1] = 0;
LABEL_9:
  *v18 = v14;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      break;
    v9 = *(_QWORD *)(v7 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
    {
      v12 = v52;
      v11 = v54;
      ++v10;
      goto LABEL_6;
    }
  }
LABEL_11:
  v56 = 0x200000000LL;
  v55 = (__int64 *)v57;
  if ( v45 )
  {
    v19 = 0;
    v20 = 2;
    v21 = 0;
    v22 = 8LL * v45;
    while ( 1 )
    {
      v33 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72) + v19);
      if ( !v54 )
        goto LABEL_21;
      v23 = (v54 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v24 = (__int64 *)(v52 + 16LL * v23);
      v25 = *v24;
      if ( v33 != *v24 )
        break;
LABEL_14:
      v26 = v24[1];
      v27 = v21 + 1;
      if ( v21 + 1 <= v20 )
        goto LABEL_15;
LABEL_22:
      v46 = v26;
      v47 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72) + v19);
      sub_C8D5F0((__int64)&v55, v57, v27, 8u, v33, v26);
      v21 = (unsigned int)v56;
      v26 = v46;
      v33 = v47;
LABEL_15:
      v55[v21] = v26;
      v28 = *(_QWORD *)(a2 + 40);
      LODWORD(v56) = v56 + 1;
      v29 = sub_2AB6E60(a1, v33, v28);
      if ( !v29 )
      {
        v38 = v55;
        v39 = (unsigned int)v56;
        goto LABEL_39;
      }
      v31 = (unsigned int)v56;
      v32 = (unsigned int)v56 + 1LL;
      if ( v32 > HIDWORD(v56) )
      {
        v48 = v29;
        sub_C8D5F0((__int64)&v55, v57, (unsigned int)v56 + 1LL, 8u, v32, v30);
        v31 = (unsigned int)v56;
        v29 = v48;
      }
      v19 += 8;
      v55[v31] = v29;
      v21 = (unsigned int)(v56 + 1);
      LODWORD(v56) = v56 + 1;
      if ( v22 == v19 )
      {
        v38 = v55;
        v39 = (unsigned int)v21;
        goto LABEL_39;
      }
      v20 = HIDWORD(v56);
    }
    v34 = 1;
    while ( v25 != -4096 )
    {
      v35 = v34 + 1;
      v23 = (v54 - 1) & (v34 + v23);
      v24 = (__int64 *)(v52 + 16LL * v23);
      v25 = *v24;
      if ( v33 == *v24 )
        goto LABEL_14;
      v34 = v35;
    }
LABEL_21:
    v27 = v21 + 1;
    v26 = 0;
    if ( v21 + 1 <= v20 )
      goto LABEL_15;
    goto LABEL_22;
  }
  v39 = 0;
  v38 = (__int64 *)v57;
LABEL_39:
  v49 = v39;
  v40 = (_QWORD *)sub_22077B0(0x98u);
  if ( v40 )
  {
    v41 = v49;
    v50 = *(__int64 **)(a2 + 48);
    if ( v50 )
    {
      sub_2AAAFA0((__int64 *)&v50);
      v41 = v49;
    }
    sub_2ABB100((__int64)v40, 25, v38, v41, a2, (__int64 *)&v50);
    sub_9C6650(&v50);
    *v40 = &unk_4A243A0;
    v40[5] = &unk_4A243E0;
    v40[12] = &unk_4A24418;
  }
  if ( v55 != (__int64 *)v57 )
    _libc_free((unsigned __int64)v55);
  sub_C7D6A0(v52, 16LL * v54, 8);
  return v40;
}
