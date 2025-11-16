// Function: sub_258E050
// Address: 0x258e050
//
__int64 __fastcall sub_258E050(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned int v11; // edx
  unsigned int v12; // ebx
  unsigned __int8 *v13; // rbx
  __int64 v14; // rax
  unsigned __int8 *v15; // rbx
  unsigned __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned int v23; // ebx
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int v27; // edx
  unsigned int v28; // r12d
  unsigned __int64 v29; // rax
  unsigned __int64 v31; // rbx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned __int64 *v37; // rax
  __int64 v38; // rbx
  bool v39; // zf
  __int64 v40; // [rsp+8h] [rbp-178h]
  char v41; // [rsp+2Eh] [rbp-152h]
  _BYTE *v42; // [rsp+30h] [rbp-150h]
  __int64 v43; // [rsp+40h] [rbp-140h]
  unsigned int v44; // [rsp+40h] [rbp-140h]
  __int64 v45; // [rsp+40h] [rbp-140h]
  char v47; // [rsp+7Dh] [rbp-103h] BYREF
  char v48; // [rsp+7Eh] [rbp-102h] BYREF
  char v49; // [rsp+7Fh] [rbp-101h] BYREF
  __int64 *v50; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v51; // [rsp+88h] [rbp-F8h]
  _QWORD v52[4]; // [rsp+90h] [rbp-F0h] BYREF
  _BYTE *v53; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v54; // [rsp+B8h] [rbp-C8h]
  _BYTE v55[48]; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD v56[2]; // [rsp+F0h] [rbp-90h] BYREF
  unsigned __int64 v57; // [rsp+100h] [rbp-80h]
  char v58[8]; // [rsp+108h] [rbp-78h] BYREF
  int v59; // [rsp+110h] [rbp-70h] BYREF
  unsigned __int64 v60; // [rsp+118h] [rbp-68h]
  int *v61; // [rsp+120h] [rbp-60h]
  int *v62; // [rsp+128h] [rbp-58h]
  __int64 v63; // [rsp+130h] [rbp-50h]
  void *v64; // [rsp+138h] [rbp-48h]
  __int16 v65; // [rsp+140h] [rbp-40h]

  v53 = v55;
  v47 = 0;
  v54 = 0x300000000LL;
  v41 = sub_2526B50(a2, (const __m128i *)(a1 + 72), a1, (__int64)&v53, 3u, &v47, 1u);
  if ( v41 )
  {
    v3 = (unsigned int)v54;
    if ( (_DWORD)v54 == 1 )
    {
      v38 = *(_QWORD *)v53;
      v39 = v38 == sub_250D070((_QWORD *)(a1 + 72));
      v3 = (unsigned int)v54;
      v41 = !v39;
    }
  }
  else
  {
    v31 = sub_250D070((_QWORD *)(a1 + 72));
    v34 = sub_2509740((_QWORD *)(a1 + 72));
    v35 = (unsigned int)v54;
    v36 = (unsigned int)v54 + 1LL;
    if ( v36 > HIDWORD(v54) )
    {
      sub_C8D5F0((__int64)&v53, v55, v36, 0x10u, v32, v33);
      v35 = (unsigned int)v54;
    }
    v37 = (unsigned __int64 *)&v53[16 * v35];
    *v37 = v31;
    v37[1] = v34;
    v3 = (unsigned int)(v54 + 1);
    LODWORD(v54) = v54 + 1;
  }
  v4 = (unsigned __int64)v53;
  v5 = *(_QWORD *)(a2 + 208);
  v61 = &v59;
  v62 = &v59;
  v6 = *(_QWORD *)(v5 + 104);
  v59 = 0;
  v56[0] = &unk_4A16DD8;
  v57 = 0xFFFFFFFF00000000LL;
  v60 = 0;
  v56[1] = &unk_4A16D78;
  v63 = 0;
  v65 = 256;
  v64 = &unk_4A16CD8;
  v42 = &v53[16 * v3];
  if ( v42 != v53 )
  {
    while ( 1 )
    {
      v13 = *(unsigned __int8 **)v4;
      v14 = *(_QWORD *)(*(_QWORD *)v4 + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
        v14 = **(_QWORD **)(v14 + 16);
      v51 = sub_AE2980(v6, *(_DWORD *)(v14 + 8) >> 8)[3];
      if ( v51 > 0x40 )
        sub_C43690((__int64)&v50, 0, 0);
      else
        v50 = 0;
      v48 = 0;
      v52[2] = &v49;
      v49 = 0;
      v52[3] = &v48;
      v52[0] = a2;
      v52[1] = a1;
      v15 = sub_BD45C0(
              v13,
              v6,
              (__int64)&v50,
              1,
              1,
              0,
              (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2589870,
              (__int64)v52);
      v16 = sub_250D2C0((unsigned __int64)v15, 0);
      v18 = sub_258DCE0(a2, v16, v17, a1, 0, 0, 1);
      if ( !v18 || a1 == v18 && v41 != 1 )
      {
        v45 = v18;
        v29 = sub_BD4FF0(v15, v6, &v48, v52);
        v20 = v45;
        v9 = v29;
        HIBYTE(v65) = v65;
      }
      else
      {
        v43 = v18;
        v19 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v18 + 48LL))(v18, v16, v40);
        v20 = v43;
        v9 = *(unsigned int *)(v19 + 20);
        v65 &= *(_WORD *)(v19 + 80);
      }
      if ( v51 <= 0x40 )
      {
        v7 = 0;
        if ( v51 )
        {
          v8 = 0;
          v7 = (__int64)((_QWORD)v50 << (64 - (unsigned __int8)v51)) >> (64 - (unsigned __int8)v51);
          if ( v7 >= 0 )
            v8 = (__int64)((_QWORD)v50 << (64 - (unsigned __int8)v51)) >> (64 - (unsigned __int8)v51);
          v9 -= v8;
        }
      }
      else
      {
        v7 = *v50;
        v21 = 0;
        if ( *v50 >= 0 )
          v21 = *v50;
        v9 -= v21;
      }
      v10 = 0;
      if ( v9 >= 0 )
        v10 = v9;
      v11 = v57;
      v12 = v10;
      if ( HIDWORD(v57) <= v10 )
        v12 = HIDWORD(v57);
      if ( v12 < (unsigned int)v57 )
        v12 = v57;
      HIDWORD(v57) = v12;
      if ( a1 != v20 )
        goto LABEL_17;
      if ( v41 )
        break;
      v22 = (__int64)v61;
      if ( v10 >= v12 )
        v12 = v10;
      if ( v10 >= (unsigned int)v57 )
        v11 = v10;
      v57 = __PAIR64__(v12, v11);
      v23 = v11;
      if ( v61 != &v59 )
      {
        v44 = v11;
        v24 = v11;
        do
        {
          v26 = *(_QWORD *)(v22 + 32);
          if ( v24 < v26 )
            break;
          v25 = *(_QWORD *)(v22 + 40) + v26;
          if ( v24 < v25 )
            v24 = v25;
          v22 = sub_220EEE0(v22);
        }
        while ( (int *)v22 != &v59 );
        v27 = v24;
        v23 = v44;
        if ( v44 < v27 )
          v23 = v27;
      }
      LODWORD(v57) = v23;
      HIDWORD(v57) = v23;
      HIBYTE(v65) = v65;
      sub_969240((__int64 *)&v50);
      if ( !v23 )
      {
LABEL_46:
        *(_DWORD *)(a1 + 108) = *(_DWORD *)(a1 + 104);
        *(_BYTE *)(a1 + 169) = *(_BYTE *)(a1 + 168);
        v28 = 0;
        goto LABEL_51;
      }
LABEL_18:
      v4 += 16LL;
      if ( v42 == (_BYTE *)v4 )
        goto LABEL_50;
    }
    if ( v7 > 0 )
    {
      HIDWORD(v57) = v57;
      v12 = v57;
      HIBYTE(v65) = v65;
    }
LABEL_17:
    sub_969240((__int64 *)&v50);
    if ( !v12 )
      goto LABEL_46;
    goto LABEL_18;
  }
LABEL_50:
  v28 = sub_25538A0(a1 + 88, (__int64)v56);
LABEL_51:
  v56[0] = &unk_4A16DD8;
  sub_255C230((__int64)v58, v60);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  return v28;
}
