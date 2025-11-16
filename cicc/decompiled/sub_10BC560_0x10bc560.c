// Function: sub_10BC560
// Address: 0x10bc560
//
__int64 __fastcall sub_10BC560(__int64 a1, __int64 a2, char a3, char a4, __int64 a5)
{
  __int64 v9; // r13
  _BYTE *v10; // r11
  int v11; // r8d
  __int64 v12; // rdx
  __int64 v13; // rdi
  _BYTE *v14; // r9
  __int64 v15; // rdi
  int v16; // esi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 result; // rax
  _BYTE *v20; // rax
  unsigned int v21; // eax
  char v22; // al
  _BYTE *v23; // r9
  char v24; // cl
  unsigned int v25; // eax
  __int64 v26; // rdi
  __int64 v27; // r14
  __int64 v28; // r15
  char v29; // al
  _QWORD *v30; // rax
  _BYTE *v31; // r9
  __int64 v32; // r14
  unsigned int *v33; // r15
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // r13
  unsigned int *v37; // r14
  __int64 v38; // rdx
  unsigned int v39; // esi
  char v40; // [rsp+4h] [rbp-10Ch]
  char v41; // [rsp+4h] [rbp-10Ch]
  char v42; // [rsp+4h] [rbp-10Ch]
  _BYTE *v43; // [rsp+8h] [rbp-108h]
  _BYTE *v44; // [rsp+8h] [rbp-108h]
  char v45; // [rsp+8h] [rbp-108h]
  _BYTE *v46; // [rsp+8h] [rbp-108h]
  int v47; // [rsp+10h] [rbp-100h]
  int v48; // [rsp+10h] [rbp-100h]
  _BYTE *v49; // [rsp+10h] [rbp-100h]
  _BYTE *v50; // [rsp+10h] [rbp-100h]
  _BYTE *v51; // [rsp+10h] [rbp-100h]
  __int64 v52; // [rsp+10h] [rbp-100h]
  _BYTE *v53; // [rsp+18h] [rbp-F8h]
  __int64 v54; // [rsp+18h] [rbp-F8h]
  __int64 v55; // [rsp+18h] [rbp-F8h]
  _BYTE *v56; // [rsp+18h] [rbp-F8h]
  _BYTE *v57; // [rsp+18h] [rbp-F8h]
  _QWORD v58[2]; // [rsp+20h] [rbp-F0h] BYREF
  const void *v59; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v60; // [rsp+38h] [rbp-D8h]
  const void *v61; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v62; // [rsp+48h] [rbp-C8h]
  char v63[32]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v64; // [rsp+70h] [rbp-A0h]
  _BYTE v65[32]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v66; // [rsp+A0h] [rbp-70h]
  _BYTE v67[32]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v68; // [rsp+D0h] [rbp-40h]

  v9 = *(_QWORD *)(a1 - 64);
  v10 = *(_BYTE **)(a2 - 64);
  v53 = *(_BYTE **)(a2 - 32);
  v11 = *(_WORD *)(a1 + 2) & 0x3F;
  if ( a3 )
  {
    v41 = a4;
    v44 = *(_BYTE **)(a2 - 64);
    v48 = sub_B52870(v11);
    v21 = sub_B52870(*(_WORD *)(a2 + 2) & 0x3F);
    v11 = v48;
    v10 = v44;
    a4 = v41;
    v12 = v21;
  }
  else
  {
    v12 = *(_WORD *)(a2 + 2) & 0x3F;
  }
  if ( v11 != 32 )
    return 0;
  v13 = *(_QWORD *)(a1 - 32);
  v14 = (_BYTE *)(v13 + 24);
  if ( *(_BYTE *)v13 != 17 )
  {
    v40 = a4;
    v43 = v10;
    v47 = v12;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17 > 1 )
      return 0;
    if ( *(_BYTE *)v13 > 0x15u )
      return 0;
    v20 = sub_AD7630(v13, 1, v12);
    if ( !v20 || *v20 != 17 )
      return 0;
    a4 = v40;
    v10 = v43;
    v14 = v20 + 24;
    LODWORD(v12) = v47;
  }
  v15 = *(_QWORD *)(v9 + 8);
  v16 = *(unsigned __int8 *)(v15 + 8);
  if ( (unsigned int)(v16 - 17) <= 1 )
    LOBYTE(v16) = *(_BYTE *)(**(_QWORD **)(v15 + 16) + 8LL);
  if ( (_BYTE)v16 != 12 )
    return 0;
  v17 = *(_QWORD *)(a1 + 16);
  if ( !v17 || *(_QWORD *)(v17 + 8) )
  {
    v18 = *(_QWORD *)(a2 + 16);
    if ( !v18 || *(_QWORD *)(v18 + 8) )
      return 0;
  }
  v58[0] = v9;
  v58[1] = v14;
  if ( (_DWORD)v12 != 36 )
  {
    v45 = a4;
    v49 = v14;
    if ( (_DWORD)v12 == 34 )
    {
      v22 = sub_10BAE90(v58, v10);
      v23 = v49;
      v24 = v45;
      if ( v22 )
        goto LABEL_22;
    }
    return 0;
  }
  v42 = a4;
  v46 = v14;
  v50 = v10;
  v29 = sub_10BAE90(v58, v53);
  v23 = v46;
  v24 = v42;
  if ( !v29 )
    return 0;
  v53 = v50;
LABEL_22:
  if ( v24 )
  {
    v51 = v23;
    v66 = 257;
    v68 = 257;
    v30 = sub_BD2C40(72, unk_3F10A14);
    v31 = v51;
    v32 = (__int64)v30;
    if ( v30 )
    {
      sub_B549F0((__int64)v30, (__int64)v53, (__int64)v67, 0, 0);
      v31 = v51;
    }
    v56 = v31;
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
      *(_QWORD *)(a5 + 88),
      v32,
      v65,
      *(_QWORD *)(a5 + 56),
      *(_QWORD *)(a5 + 64));
    v23 = v56;
    v33 = *(unsigned int **)a5;
    v52 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
    if ( *(_QWORD *)a5 != v52 )
    {
      do
      {
        v34 = *((_QWORD *)v33 + 1);
        v35 = *v33;
        v33 += 4;
        v57 = v23;
        sub_B99FD0(v32, v35, v34);
        v23 = v57;
      }
      while ( (unsigned int *)v52 != v33 );
    }
    v53 = (_BYTE *)v32;
  }
  v66 = 257;
  v64 = 257;
  v60 = *((_DWORD *)v23 + 2);
  if ( v60 > 0x40 )
    sub_C43780((__int64)&v59, (const void **)v23);
  else
    v59 = *(const void **)v23;
  sub_C46A40((__int64)&v59, 1);
  v25 = v60;
  v60 = 0;
  v62 = v25;
  v26 = *(_QWORD *)(v9 + 8);
  v61 = v59;
  v27 = sub_AD8D80(v26, (__int64)&v61);
  v28 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a5 + 80) + 32LL))(
          *(_QWORD *)(a5 + 80),
          15,
          v9,
          v27,
          0,
          0);
  if ( !v28 )
  {
    v68 = 257;
    v28 = sub_B504D0(15, v9, v27, (__int64)v67, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
      *(_QWORD *)(a5 + 88),
      v28,
      v63,
      *(_QWORD *)(a5 + 56),
      *(_QWORD *)(a5 + 64));
    v36 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
    v37 = *(unsigned int **)a5;
    if ( *(_QWORD *)a5 != v36 )
    {
      do
      {
        v38 = *((_QWORD *)v37 + 1);
        v39 = *v37;
        v37 += 4;
        sub_B99FD0(v28, v39, v38);
      }
      while ( (unsigned int *)v36 != v37 );
    }
  }
  result = sub_92B530((unsigned int **)a5, 35 - ((unsigned int)(a3 == 0) - 1), v28, v53, (__int64)v65);
  if ( v62 > 0x40 && v61 )
  {
    v54 = result;
    j_j___libc_free_0_0(v61);
    result = v54;
  }
  if ( v60 > 0x40 )
  {
    if ( v59 )
    {
      v55 = result;
      j_j___libc_free_0_0(v59);
      return v55;
    }
  }
  return result;
}
