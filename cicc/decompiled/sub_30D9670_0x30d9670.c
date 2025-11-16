// Function: sub_30D9670
// Address: 0x30d9670
//
__int64 __fastcall sub_30D9670(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 v9; // r15
  unsigned int v10; // r13d
  int v11; // ecx
  __int64 v12; // rsi
  int v13; // ecx
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r14
  unsigned int v18; // edx
  unsigned __int64 v19; // rax
  unsigned int v20; // edx
  unsigned int v21; // r14d
  unsigned int v22; // eax
  unsigned int v23; // r15d
  __int64 v24; // r14
  int v25; // eax
  __int64 v26; // rsi
  __int64 v27; // r14
  _BYTE *v28; // rdi
  __int64 v29; // rsi
  int v31; // ecx
  unsigned int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // r13
  int v36; // eax
  int v37; // r8d
  char v38; // al
  __int64 v39; // rax
  int v40; // edi
  unsigned __int64 v41; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v42; // [rsp+18h] [rbp-68h]
  unsigned __int64 v43; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+28h] [rbp-58h]
  __int64 v45; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v46; // [rsp+38h] [rbp-48h] BYREF
  unsigned int v47; // [rsp+40h] [rbp-40h]

  v8 = *(_QWORD *)(a2 - 64);
  v9 = *(_QWORD *)(a2 - 32);
  v10 = sub_30D92D0(a1, a2, a3, a4, a5, a6);
  if ( (_BYTE)v10 || *(_BYTE *)a2 == 83 )
    return v10;
  v11 = *(_DWORD *)(a1 + 256);
  v42 = 1;
  v41 = 0;
  v12 = *(_QWORD *)(a1 + 240);
  v44 = 1;
  v43 = 0;
  if ( !v11 )
    goto LABEL_32;
  v13 = v11 - 1;
  v14 = v13 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v15 = (__int64 *)(v12 + 32LL * v14);
  v16 = *v15;
  if ( v8 != *v15 )
  {
    v36 = 1;
    while ( v16 != -4096 )
    {
      v37 = v36 + 1;
      v14 = v13 & (v36 + v14);
      v15 = (__int64 *)(v12 + 32LL * v14);
      v16 = *v15;
      if ( v8 == *v15 )
        goto LABEL_5;
      v36 = v37;
    }
LABEL_32:
    v41 = 0;
    v42 = 1;
    goto LABEL_10;
  }
LABEL_5:
  v17 = v15[1];
  v45 = v17;
  v18 = *((_DWORD *)v15 + 6);
  v47 = v18;
  if ( v18 > 0x40 )
  {
    sub_C43780((__int64)&v46, (const void **)v15 + 2);
    v17 = v45;
    v19 = v46;
    v18 = v47;
  }
  else
  {
    v19 = v15[2];
  }
  v41 = v19;
  v42 = v18;
  if ( v17 )
  {
    sub_30D74B0((__int64)&v45, a1 + 232, v9);
    v43 = v46;
    v44 = v47;
    LOBYTE(v20) = v45 != 0 && v45 == v17;
    v21 = v20;
    if ( (_BYTE)v20 )
    {
      v38 = sub_B532C0((__int64)&v41, &v43, *(_WORD *)(a2 + 2) & 0x3F);
      v39 = sub_AD64A0(*(_QWORD *)(a2 + 8), v38);
      v45 = a2;
      *sub_30D9190(a1 + 136, &v45) = v39;
      v10 = v21;
      ++*(_DWORD *)(a1 + 636);
      goto LABEL_23;
    }
  }
LABEL_10:
  LOBYTE(v22) = sub_B52830(*(_WORD *)(a2 + 2) & 0x3F);
  v23 = v22;
  if ( !(_BYTE)v22 )
    goto LABEL_20;
  v24 = *(_QWORD *)(a2 - 64);
  if ( **(_BYTE **)(a2 - 32) != 20 )
  {
    v29 = sub_30D1740(a1, *(_QWORD *)(a2 - 64));
    if ( !v29 )
      goto LABEL_23;
    goto LABEL_31;
  }
  if ( *(_BYTE *)v24 == 22 && (unsigned __int8)sub_B49B80(*(_QWORD *)(a1 + 96), *(_DWORD *)(v24 + 32), 43) )
  {
LABEL_34:
    v34 = *(_QWORD *)(a2 + 8);
    if ( (*(_WORD *)(a2 + 2) & 0x3F) == 0x21 )
      v35 = sub_AD6400(v34);
    else
      v35 = sub_AD6450(v34);
    v45 = a2;
    *sub_30D9190(a1 + 136, &v45) = v35;
    v10 = v23;
    goto LABEL_23;
  }
  v25 = *(_DWORD *)(a1 + 192);
  v26 = *(_QWORD *)(a1 + 176);
  if ( v25 )
  {
    v31 = v25 - 1;
    v32 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v33 = *(_QWORD *)(v26 + 16LL * v32);
    if ( v33 == v24 )
      goto LABEL_34;
    v40 = 1;
    while ( v33 != -4096 )
    {
      v32 = v31 & (v40 + v32);
      v33 = *(_QWORD *)(v26 + 16LL * v32);
      if ( v33 == v24 )
        goto LABEL_34;
      ++v40;
    }
  }
  v27 = *(_QWORD *)(a2 + 16);
  if ( v27 )
  {
    while ( 1 )
    {
      v28 = *(_BYTE **)(v27 + 24);
      if ( *v28 > 0x1Cu && ((v28[7] & 0x20) == 0 || !sub_B91C10((__int64)v28, 14)) )
        break;
      v27 = *(_QWORD *)(v27 + 8);
      if ( !v27 )
        goto LABEL_44;
    }
LABEL_20:
    v29 = sub_30D1740(a1, *(_QWORD *)(a2 - 64));
    if ( !v29 )
      goto LABEL_23;
    if ( **(_BYTE **)(a2 - 32) == 20 )
    {
      v10 = 1;
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 184LL))(a1, v29);
      goto LABEL_23;
    }
LABEL_31:
    sub_30D1890(a1, v29);
    goto LABEL_23;
  }
LABEL_44:
  v10 = v23;
LABEL_23:
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  return v10;
}
