// Function: sub_2AC9810
// Address: 0x2ac9810
//
void __fastcall sub_2AC9810(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 v5; // dx
  __int64 v6; // r15
  char v7; // r12
  __int64 v8; // rbx
  __int16 v9; // ax
  __int64 v10; // rdi
  __int64 v11; // r9
  const char *v12; // rbx
  unsigned __int64 v13; // rax
  int v14; // ecx
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rcx
  int v19; // r11d
  char *v20; // rdi
  __int64 v21; // r9
  unsigned int v22; // edx
  char *v23; // rax
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rax
  __int16 v27; // cx
  __int64 v28; // r15
  __int64 v29; // r12
  unsigned __int8 *v30; // r12
  unsigned __int8 *v31; // rax
  unsigned int v32; // esi
  int v33; // edx
  int v34; // eax
  __int64 v35; // rdi
  __int64 v36; // r8
  unsigned int v37; // esi
  __int64 *v38; // rcx
  __int64 v39; // r9
  int v40; // ecx
  int v41; // r11d
  unsigned __int64 v42; // r8
  __int64 v43; // [rsp+8h] [rbp-148h]
  __int64 v45; // [rsp+30h] [rbp-120h]
  char v46; // [rsp+38h] [rbp-118h]
  __int64 v47; // [rsp+38h] [rbp-118h]
  __int64 v48; // [rsp+58h] [rbp-F8h] BYREF
  const char *v49[4]; // [rsp+60h] [rbp-F0h] BYREF
  char v50; // [rsp+80h] [rbp-D0h]
  char v51; // [rsp+81h] [rbp-CFh]
  _BYTE *v52; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+98h] [rbp-B8h]
  _BYTE v54[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v55; // [rsp+C0h] [rbp-90h]
  __int64 v56; // [rsp+C8h] [rbp-88h]
  __int16 v57; // [rsp+D0h] [rbp-80h]
  __int64 v58; // [rsp+D8h] [rbp-78h]
  void **v59; // [rsp+E0h] [rbp-70h]
  void **v60; // [rsp+E8h] [rbp-68h]
  __int64 v61; // [rsp+F0h] [rbp-60h]
  int v62; // [rsp+F8h] [rbp-58h]
  __int16 v63; // [rsp+FCh] [rbp-54h]
  char v64; // [rsp+FEh] [rbp-52h]
  __int64 v65; // [rsp+100h] [rbp-50h]
  __int64 v66; // [rsp+108h] [rbp-48h]
  void *v67; // [rsp+110h] [rbp-40h] BYREF
  void *v68; // [rsp+118h] [rbp-38h] BYREF

  v45 = *(_QWORD *)(*(_QWORD *)(a1 + 376) + 72LL);
  v6 = sub_AA5190(*(_QWORD *)(a1 + 456));
  if ( v6 )
  {
    v7 = v5;
    v46 = HIBYTE(v5);
  }
  else
  {
    v46 = 0;
    v7 = 0;
  }
  v8 = *(_QWORD *)(a1 + 456);
  v58 = sub_AA48A0(v8);
  v59 = &v67;
  v60 = &v68;
  v53 = 0x200000000LL;
  v67 = &unk_49DA100;
  v52 = v54;
  v61 = 0;
  v68 = &unk_49DA0B0;
  LOBYTE(v9) = v7;
  v62 = 0;
  HIBYTE(v9) = v46;
  v63 = 512;
  v64 = 7;
  v65 = 0;
  v66 = 0;
  v55 = v8;
  v56 = v6;
  v57 = v9;
  if ( v6 != v8 + 48 )
  {
    v10 = v6 - 24;
    if ( !v6 )
      v10 = 0;
    v49[0] = *(const char **)sub_B46C60(v10);
    if ( !v49[0] || (sub_2AAAFA0((__int64 *)v49), (v12 = v49[0]) == 0) )
    {
      sub_93FB40((__int64)&v52, 0);
LABEL_50:
      sub_9C6650(v49);
      goto LABEL_13;
    }
    v13 = (unsigned __int64)v52;
    v14 = v53;
    v15 = &v52[16 * (unsigned int)v53];
    if ( v52 == (_BYTE *)v15 )
    {
LABEL_51:
      if ( (unsigned int)v53 >= (unsigned __int64)HIDWORD(v53) )
      {
        v42 = (unsigned int)v53 + 1LL;
        if ( HIDWORD(v53) < v42 )
        {
          sub_C8D5F0((__int64)&v52, v54, v42, 0x10u, v42, v11);
          v15 = &v52[16 * (unsigned int)v53];
        }
        *v15 = 0;
        v15[1] = v12;
        LODWORD(v53) = v53 + 1;
      }
      else
      {
        if ( v15 )
        {
          *(_DWORD *)v15 = 0;
          v15[1] = v12;
          v14 = v53;
        }
        LODWORD(v53) = v14 + 1;
      }
      goto LABEL_50;
    }
    while ( *(_DWORD *)v13 )
    {
      v13 += 16LL;
      if ( v15 == (_QWORD *)v13 )
        goto LABEL_51;
    }
    *(const char **)(v13 + 8) = v49[0];
    sub_9C6650(v49);
  }
LABEL_13:
  v16 = *(_QWORD *)(a1 + 376);
  v43 = a1 + 424;
  v17 = *(_QWORD *)(v16 + 160);
  v47 = v17 + 88LL * *(unsigned int *)(v16 + 168);
  if ( v17 != v47 )
  {
    while ( 1 )
    {
      v25 = *(_QWORD *)v17;
      v48 = *(_QWORD *)v17;
      v26 = *(_QWORD *)(v17 + 40);
      v27 = *(_WORD *)(v26 + 24);
      if ( !v27 )
      {
        v28 = *(_QWORD *)(v26 + 32);
        goto LABEL_19;
      }
      if ( v27 == 15 )
      {
        v28 = *(_QWORD *)(v26 - 8);
        goto LABEL_19;
      }
      v35 = *(unsigned int *)(a2 + 24);
      v36 = *(_QWORD *)(a2 + 8);
      if ( !(_DWORD)v35 )
        goto LABEL_57;
      v37 = (v35 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v38 = (__int64 *)(v36 + 16LL * v37);
      v39 = *v38;
      if ( v26 != *v38 )
        break;
LABEL_48:
      v28 = v38[1];
LABEL_19:
      v29 = a3;
      if ( v25 != v45 )
      {
        v30 = *(unsigned __int8 **)(v17 + 48);
        if ( v30 && (unsigned __int8)sub_920620(*(_QWORD *)(v17 + 48)) )
          v62 = sub_B45210((__int64)v30);
        v31 = sub_2ABE630((__int64)&v52, a3, *(_QWORD *)(v17 + 24), v28, *(_DWORD *)(v17 + 32), v30);
        v51 = 1;
        v29 = (__int64)v31;
        v50 = 3;
        v49[0] = "ind.end";
        sub_BD6B50(v31, v49);
      }
      v32 = *(_DWORD *)(a1 + 448);
      if ( !v32 )
      {
        ++*(_QWORD *)(a1 + 424);
        v49[0] = 0;
        goto LABEL_26;
      }
      v18 = v48;
      v19 = 1;
      v20 = 0;
      v21 = *(_QWORD *)(a1 + 432);
      v22 = (v32 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v23 = (char *)(v21 + 16LL * v22);
      v24 = *(_QWORD *)v23;
      if ( v48 == *(_QWORD *)v23 )
      {
LABEL_16:
        v17 += 88;
        *((_QWORD *)v23 + 1) = v29;
        if ( v47 == v17 )
          goto LABEL_41;
      }
      else
      {
        while ( v24 != -4096 )
        {
          if ( v24 == -8192 && !v20 )
            v20 = v23;
          v22 = (v32 - 1) & (v19 + v22);
          v23 = (char *)(v21 + 16LL * v22);
          v24 = *(_QWORD *)v23;
          if ( v48 == *(_QWORD *)v23 )
            goto LABEL_16;
          ++v19;
        }
        if ( !v20 )
          v20 = v23;
        v34 = *(_DWORD *)(a1 + 440);
        ++*(_QWORD *)(a1 + 424);
        v33 = v34 + 1;
        v49[0] = v20;
        if ( 4 * (v34 + 1) < 3 * v32 )
        {
          if ( v32 - *(_DWORD *)(a1 + 444) - v33 > v32 >> 3 )
            goto LABEL_38;
          goto LABEL_27;
        }
LABEL_26:
        v32 *= 2;
LABEL_27:
        sub_2AC9660(v43, v32);
        sub_2ABF280(v43, &v48, v49);
        v18 = v48;
        v20 = (char *)v49[0];
        v33 = *(_DWORD *)(a1 + 440) + 1;
LABEL_38:
        *(_DWORD *)(a1 + 440) = v33;
        if ( *(_QWORD *)v20 != -4096 )
          --*(_DWORD *)(a1 + 444);
        *(_QWORD *)v20 = v18;
        v17 += 88;
        *((_QWORD *)v20 + 1) = 0;
        *((_QWORD *)v20 + 1) = v29;
        if ( v47 == v17 )
          goto LABEL_41;
      }
    }
    v40 = 1;
    while ( v39 != -4096 )
    {
      v41 = v40 + 1;
      v37 = (v35 - 1) & (v40 + v37);
      v38 = (__int64 *)(v36 + 16LL * v37);
      v39 = *v38;
      if ( v26 == *v38 )
        goto LABEL_48;
      v40 = v41;
    }
LABEL_57:
    v38 = (__int64 *)(v36 + 16 * v35);
    goto LABEL_48;
  }
LABEL_41:
  nullsub_61();
  v67 = &unk_49DA100;
  nullsub_63();
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
}
