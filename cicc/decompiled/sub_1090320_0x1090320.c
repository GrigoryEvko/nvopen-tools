// Function: sub_1090320
// Address: 0x1090320
//
__int64 __fastcall sub_1090320(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v12; // r12
  __int64 v13; // rax
  __int16 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r9
  __int64 v20; // r14
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 result; // rax
  __int64 v28; // r15
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rsi
  unsigned int v32; // ecx
  __int64 *v33; // rdx
  __int64 v34; // r8
  int v35; // edx
  __int64 v36; // rbx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rbx
  int v42; // edx
  int v43; // edx
  void *v44; // rax
  void *v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // rax
  int v48; // r10d
  int v49; // r9d
  __int64 v50; // rdx
  __int64 v51; // rsi
  void *v52; // rax
  __int64 v53; // [rsp+8h] [rbp-88h]
  char v54; // [rsp+14h] [rbp-7Ch]
  __int16 v55; // [rsp+16h] [rbp-7Ah]
  __int64 v57; // [rsp+20h] [rbp-70h]
  __int64 v59; // [rsp+30h] [rbp-60h] BYREF
  __int64 v60; // [rsp+38h] [rbp-58h] BYREF
  __int64 v61; // [rsp+40h] [rbp-50h] BYREF
  int v62; // [rsp+48h] [rbp-48h]
  __int64 v63; // [rsp+50h] [rbp-40h] BYREF
  int v64; // [rsp+58h] [rbp-38h]

  v12 = *(_QWORD *)(a7 + 16);
  v13 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a2 + 8) + 64LL))(
          *(_QWORD *)(a2 + 8),
          *(unsigned int *)(a4 + 12));
  v14 = (*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64, _QWORD))(**(_QWORD **)(a1 + 184) + 24LL))(
          *(_QWORD *)(a1 + 184),
          &a7,
          a4,
          *(_DWORD *)(v13 + 16) & 1);
  v55 = __ROL2__(v14, 8);
  v54 = HIBYTE(v14);
  v59 = sub_108B910(v12);
  HIDWORD(v57) = *(_DWORD *)(a4 + 8) + sub_E5C2C0(a2, a3);
  v15 = *(unsigned int *)(a1 + 304);
  v16 = *(_QWORD *)(a1 + 288);
  if ( !(_DWORD)v15 )
    goto LABEL_31;
  v17 = (v15 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v18 = (__int64 *)(v16 + 16LL * v17);
  v19 = *v18;
  if ( v12 != *v18 )
  {
    v42 = 1;
    while ( v19 != -4096 )
    {
      v48 = v42 + 1;
      v17 = (v15 - 1) & (v42 + v17);
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v12 == *v18 )
        goto LABEL_3;
      v42 = v48;
    }
    goto LABEL_31;
  }
LABEL_3:
  if ( v18 == (__int64 *)(v16 + 16 * v15) )
  {
LABEL_31:
    v63 = *(_QWORD *)(v59 + 152);
    LODWORD(v57) = *(_DWORD *)sub_10900D0(a1 + 280, &v63);
    goto LABEL_5;
  }
  LODWORD(v57) = *((_DWORD *)v18 + 2);
LABEL_5:
  v20 = a1 + 248;
  if ( (unsigned __int8)(v14 - 32) <= 3u || !(_BYTE)v14 )
  {
    v63 = v59;
    if ( *(_BYTE *)(v59 + 180) )
    {
      v22 = sub_E5C4C0(a2, v12);
    }
    else if ( *(_QWORD *)v12
           || (*(_BYTE *)(v12 + 9) & 0x70) == 0x20
           && *(char *)(v12 + 8) >= 0
           && (*(_BYTE *)(v12 + 8) |= 8u, v45 = sub_E807D0(*(_QWORD *)(v12 + 24)), (*(_QWORD *)v12 = v45) != 0) )
    {
      v21 = *(_QWORD *)(*sub_108F450(a1 + 248, &v63) + 16LL);
      v22 = v21 + sub_E5C4C0(a2, v12);
    }
    else
    {
      v22 = *(_QWORD *)(*sub_108F450(a1 + 248, &v63) + 16LL);
    }
    *a5 = a9 + v22;
    goto LABEL_11;
  }
  switch ( (_BYTE)v14 )
  {
    case 0x24:
      goto LABEL_50;
    case 3:
    case 0x31:
      if ( *(_BYTE *)(v59 + 149) )
      {
        v46 = sub_108F450(a1 + 248, &v59);
        v47 = *(_QWORD *)(*v46 + 16LL) + a9 - *(_QWORD *)(*(_QWORD *)(a1 + 728) + 16LL);
        if ( (_BYTE)v14 == 3 && (__int16)v47 != v47 )
          v47 = (__int16)v47;
        goto LABEL_54;
      }
LABEL_50:
      *a5 = 0;
      break;
    case 0x1A:
      v61 = *(_QWORD *)(a3 + 8);
      v53 = *(_QWORD *)(*sub_108F450(a1 + 248, &v61) + 16LL);
      v63 = v59;
      if ( *(_BYTE *)(v59 + 180) )
      {
        v50 = sub_E5C4C0(a2, v12);
      }
      else if ( *(_QWORD *)v12
             || (*(_BYTE *)(v12 + 9) & 0x70) == 0x20
             && *(char *)(v12 + 8) >= 0
             && (*(_BYTE *)(v12 + 8) |= 8u, v52 = sub_E807D0(*(_QWORD *)(v12 + 24)), (*(_QWORD *)v12 = v52) != 0) )
      {
        v51 = *(_QWORD *)(*sub_108F450(a1 + 248, &v63) + 16LL);
        v50 = v51 + sub_E5C4C0(a2, v12);
      }
      else
      {
        v50 = *(_QWORD *)(*sub_108F450(a1 + 248, &v63) + 16LL);
      }
      v47 = v50 + a9 - HIDWORD(v57) - v53;
LABEL_54:
      *a5 = v47;
      break;
    case 0xF:
      HIDWORD(v57) = 0;
      *a5 = 0;
      break;
  }
LABEL_11:
  v60 = *(_QWORD *)(a3 + 8);
  v24 = *sub_108F450(a1 + 248, &v60);
  v63 = v57;
  LOWORD(v64) = v55;
  v25 = *(unsigned int *)(v24 + 72);
  if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(v24 + 76) )
  {
    sub_C8D5F0(v24 + 64, (const void *)(v24 + 80), v25 + 1, 0xCu, v25 + 1, v23);
    v25 = *(unsigned int *)(v24 + 72);
  }
  v26 = *(_QWORD *)(v24 + 64) + 12 * v25;
  *(_QWORD *)v26 = v63;
  *(_DWORD *)(v26 + 8) = v64;
  result = a8;
  ++*(_DWORD *)(v24 + 72);
  if ( result )
  {
    v28 = *(_QWORD *)(result + 16);
    if ( v12 == v28 )
      sub_C64ED0("relocation for opposite term is not yet supported", 1u);
    v29 = sub_108B910(*(_QWORD *)(result + 16));
    if ( v59 == v29 )
      sub_C64ED0("relocation for paired relocatable term is not yet supported", 1u);
    v30 = *(unsigned int *)(a1 + 304);
    v31 = *(_QWORD *)(a1 + 288);
    if ( (_DWORD)v30 )
    {
      v32 = (v30 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v33 = (__int64 *)(v31 + 16LL * v32);
      v34 = *v33;
      if ( v28 == *v33 )
      {
LABEL_18:
        if ( v33 != (__int64 *)(v31 + 16 * v30) )
        {
          v35 = *((_DWORD *)v33 + 2);
LABEL_20:
          LODWORD(v57) = v35;
          HIWORD(v62) = 0;
          v36 = *sub_108F450(a1 + 248, &v60);
          BYTE1(v62) = 1;
          v61 = v57;
          LOBYTE(v62) = v54;
          v63 = v57;
          v64 = v62;
          v39 = *(unsigned int *)(v36 + 72);
          if ( v39 + 1 > (unsigned __int64)*(unsigned int *)(v36 + 76) )
          {
            sub_C8D5F0(v36 + 64, (const void *)(v36 + 80), v39 + 1, 0xCu, v37, v38);
            v39 = *(unsigned int *)(v36 + 72);
          }
          v40 = *(_QWORD *)(v36 + 64) + 12 * v39;
          *(_QWORD *)v40 = v63;
          *(_DWORD *)(v40 + 8) = v64;
          ++*(_DWORD *)(v36 + 72);
          v63 = v29;
          if ( *(_BYTE *)(v29 + 180) )
          {
            result = sub_E5C4C0(a2, v28);
          }
          else if ( *(_QWORD *)v28
                 || (*(_BYTE *)(v28 + 9) & 0x70) == 0x20
                 && *(char *)(v28 + 8) >= 0
                 && (*(_BYTE *)(v28 + 8) |= 8u, v44 = sub_E807D0(*(_QWORD *)(v28 + 24)), (*(_QWORD *)v28 = v44) != 0) )
          {
            v41 = *(_QWORD *)(*sub_108F450(v20, &v63) + 16LL);
            result = v41 + sub_E5C4C0(a2, v28);
          }
          else
          {
            result = *(_QWORD *)(*sub_108F450(v20, &v63) + 16LL);
          }
          *a5 -= result;
          return result;
        }
      }
      else
      {
        v43 = 1;
        while ( v34 != -4096 )
        {
          v49 = v43 + 1;
          v32 = (v30 - 1) & (v43 + v32);
          v33 = (__int64 *)(v31 + 16LL * v32);
          v34 = *v33;
          if ( v28 == *v33 )
            goto LABEL_18;
          v43 = v49;
        }
      }
    }
    v63 = *(_QWORD *)(v29 + 152);
    v35 = *(_DWORD *)sub_10900D0(a1 + 280, &v63);
    goto LABEL_20;
  }
  return result;
}
