// Function: sub_30D43C0
// Address: 0x30d43c0
//
__int64 __fastcall sub_30D43C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned __int8 **v4; // r12
  signed __int64 v5; // rbx
  unsigned __int8 *v6; // rsi
  __int64 v7; // rdx
  unsigned int v8; // r8d
  int v9; // eax
  bool v10; // al
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // r13
  __int64 v13; // r9
  __int64 v14; // rcx
  _QWORD *v15; // r15
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rbx
  unsigned __int8 v20; // dl
  unsigned __int64 v22; // r8
  char v23; // r15
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  int v28; // ecx
  __int64 v29; // rdi
  int v30; // ecx
  unsigned int v31; // edx
  unsigned __int8 **v32; // rax
  unsigned __int8 *v33; // r8
  unsigned __int64 v34; // r13
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  int v40; // eax
  int v41; // r9d
  __int64 v44; // [rsp+18h] [rbp-88h]
  unsigned int v45; // [rsp+20h] [rbp-80h]
  unsigned int v46; // [rsp+24h] [rbp-7Ch]
  __int64 v47; // [rsp+28h] [rbp-78h]
  __int64 v48; // [rsp+28h] [rbp-78h]
  __int64 v49; // [rsp+28h] [rbp-78h]
  __int64 v50; // [rsp+30h] [rbp-70h]
  __int64 v51; // [rsp+30h] [rbp-70h]
  __int64 v52; // [rsp+30h] [rbp-70h]
  char **v53; // [rsp+38h] [rbp-68h]
  unsigned __int64 v54; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v55; // [rsp+48h] [rbp-58h]
  unsigned __int64 v56; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v57; // [rsp+58h] [rbp-48h]
  unsigned __int64 v58; // [rsp+60h] [rbp-40h] BYREF
  __int64 v59; // [rsp+68h] [rbp-38h]

  v44 = a2;
  v45 = sub_AE43F0(*(_QWORD *)(a1 + 80), *(_QWORD *)(a2 + 8));
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a2 - 8);
  else
    v3 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v4 = (unsigned __int8 **)(v3 + 32);
  v5 = sub_BB5290(a2) & 0xFFFFFFFFFFFFFFF9LL | 4;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v44 = *(_QWORD *)(a2 - 8) + 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (unsigned __int8 **)v44 == v4 )
    return 1;
  while ( 1 )
  {
    v6 = *v4;
    v7 = (__int64)*v4;
    if ( **v4 == 17 )
      goto LABEL_7;
    v28 = *(_DWORD *)(a1 + 160);
    v29 = *(_QWORD *)(a1 + 144);
    if ( !v28 )
      return 0;
    v30 = v28 - 1;
    v31 = v30 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v7 >> 4));
    v32 = (unsigned __int8 **)(v29 + 16LL * v31);
    v33 = *v32;
    if ( v6 != *v32 )
      break;
LABEL_47:
    v7 = (__int64)v32[1];
    if ( !v7 || *(_BYTE *)v7 != 17 )
      return 0;
LABEL_7:
    v8 = *(_DWORD *)(v7 + 32);
    v53 = (char **)(v7 + 24);
    if ( v8 <= 0x40 )
    {
      v10 = *(_QWORD *)(v7 + 24) == 0;
    }
    else
    {
      v46 = *(_DWORD *)(v7 + 32);
      v47 = v7;
      v9 = sub_C444A0((__int64)v53);
      v8 = v46;
      v7 = v47;
      v10 = v46 == v9;
    }
    v11 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v10 )
      goto LABEL_39;
    v13 = *(_QWORD *)(a1 + 80);
    v14 = (v5 >> 1) & 3;
    if ( !v5 )
    {
      v49 = *(_QWORD *)(a1 + 80);
      v38 = sub_BCBAE0(v11, v6, v7);
      v13 = v49;
      v22 = v38;
      goto LABEL_27;
    }
    if ( !v14 )
    {
      if ( v11 )
      {
        v15 = *(_QWORD **)(v7 + 24);
        if ( v8 > 0x40 )
          v15 = (_QWORD *)*v15;
        v16 = 16LL * (unsigned int)v15 + sub_AE4AC0(*(_QWORD *)(a1 + 80), v5 & 0xFFFFFFFFFFFFFFF8LL) + 24;
        v17 = *(_QWORD *)v16;
        LOBYTE(v16) = *(_BYTE *)(v16 + 8);
        v56 = v17;
        LOBYTE(v57) = v16;
        v18 = sub_CA1930(&v56);
        LODWORD(v59) = v45;
        if ( v45 > 0x40 )
          sub_C43690((__int64)&v58, v18, 0);
        else
          v58 = v18;
        sub_C45EE0(a3, (__int64 *)&v58);
        if ( (unsigned int)v59 > 0x40 && v58 )
          j_j___libc_free_0_0(v58);
        v19 = (v5 >> 1) & 3;
        if ( v19 == 2 )
          goto LABEL_21;
        goto LABEL_50;
      }
LABEL_62:
      v51 = *(_QWORD *)(a1 + 80);
      v37 = sub_BCBAE0(v5 & 0xFFFFFFFFFFFFFFF8LL, v6, v7);
      v13 = v51;
      v22 = v37;
      goto LABEL_27;
    }
    if ( v14 != 2 )
    {
      if ( v14 == 1 )
      {
        if ( v11 )
        {
          v35 = *(_QWORD *)(v11 + 24);
        }
        else
        {
          v52 = *(_QWORD *)(a1 + 80);
          v39 = sub_BCBAE0(0, v6, v7);
          v13 = v52;
          v35 = v39;
        }
        v36 = sub_9208B0(v13, v35);
        v59 = v25;
        v26 = (unsigned __int64)(v36 + 7) >> 3;
        goto LABEL_28;
      }
      goto LABEL_62;
    }
    v22 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v11 )
      goto LABEL_62;
LABEL_27:
    v48 = v22;
    v50 = v13;
    v23 = sub_AE5020(v13, v22);
    v24 = sub_9208B0(v50, v48);
    v59 = v25;
    v26 = ((1LL << v23) + ((unsigned __int64)(v24 + 7) >> 3) - 1) >> v23 << v23;
LABEL_28:
    v58 = v26;
    LOBYTE(v59) = v25;
    v27 = sub_CA1930(&v58);
    v55 = v45;
    if ( v45 > 0x40 )
      sub_C43690((__int64)&v54, v27, 0);
    else
      v54 = v27;
    sub_C44B10((__int64)&v56, v53, v45);
    sub_C472A0((__int64)&v58, (__int64)&v56, (__int64 *)&v54);
    sub_C45EE0(a3, (__int64 *)&v58);
    if ( (unsigned int)v59 > 0x40 && v58 )
      j_j___libc_free_0_0(v58);
    if ( v57 > 0x40 && v56 )
      j_j___libc_free_0_0(v56);
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
LABEL_39:
    if ( !v5 )
      goto LABEL_42;
    v19 = (v5 >> 1) & 3;
    if ( v19 == 2 )
    {
      if ( v11 )
        goto LABEL_21;
      goto LABEL_42;
    }
LABEL_50:
    if ( v19 == 1 && v11 )
    {
      v12 = *(_QWORD *)(v11 + 24);
LABEL_21:
      v20 = *(_BYTE *)(v12 + 8);
      if ( v20 != 16 )
        goto LABEL_43;
      goto LABEL_22;
    }
LABEL_42:
    v12 = sub_BCBAE0(v11, *v4, v7);
    v20 = *(_BYTE *)(v12 + 8);
    if ( v20 != 16 )
    {
LABEL_43:
      if ( (unsigned int)v20 - 17 > 1 )
      {
        v34 = v12 & 0xFFFFFFFFFFFFFFF9LL;
        v5 = 0;
        if ( v20 == 15 )
          v5 = v34;
      }
      else
      {
        v5 = v12 & 0xFFFFFFFFFFFFFFF9LL | 2;
      }
      goto LABEL_23;
    }
LABEL_22:
    v5 = *(_QWORD *)(v12 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
LABEL_23:
    v4 += 4;
    if ( (unsigned __int8 **)v44 == v4 )
      return 1;
  }
  v40 = 1;
  while ( v33 != (unsigned __int8 *)-4096LL )
  {
    v41 = v40 + 1;
    v31 = v30 & (v40 + v31);
    v32 = (unsigned __int8 **)(v29 + 16LL * v31);
    v33 = *v32;
    if ( v6 == *v32 )
      goto LABEL_47;
    v40 = v41;
  }
  return 0;
}
