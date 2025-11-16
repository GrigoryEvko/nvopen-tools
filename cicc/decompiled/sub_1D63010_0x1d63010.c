// Function: sub_1D63010
// Address: 0x1d63010
//
__int64 __fastcall sub_1D63010(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r15d
  __int64 v6; // r14
  bool v10; // al
  __int64 v11; // rcx
  __int64 v12; // rdx
  bool v13; // al
  __int64 v14; // r8
  __int64 v15; // r14
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r11
  unsigned int v24; // eax
  __int64 v25; // r14
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r10
  __int64 v30; // rax
  __int64 v31; // rax
  _BYTE *v32; // rdi
  __int64 v33; // rdx
  _QWORD *v34; // rsi
  __int64 v35; // rcx
  __int64 v36; // rdx
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  bool v39; // al
  __int64 v40; // [rsp+0h] [rbp-410h]
  __int64 v41; // [rsp+20h] [rbp-3F0h]
  __int64 v42; // [rsp+28h] [rbp-3E8h]
  __int64 v43; // [rsp+30h] [rbp-3E0h]
  __int64 v44; // [rsp+38h] [rbp-3D8h]
  __int64 v45; // [rsp+40h] [rbp-3D0h]
  __int64 v46; // [rsp+48h] [rbp-3C8h]
  unsigned int v47; // [rsp+48h] [rbp-3C8h]
  _QWORD v48[2]; // [rsp+50h] [rbp-3C0h] BYREF
  _OWORD v49[3]; // [rsp+60h] [rbp-3B0h] BYREF
  __int64 v50; // [rsp+90h] [rbp-380h]
  _QWORD v51[5]; // [rsp+A0h] [rbp-370h] BYREF
  unsigned int v52; // [rsp+C8h] [rbp-348h]
  __int64 v53; // [rsp+D0h] [rbp-340h]
  _OWORD *v54; // [rsp+D8h] [rbp-338h]
  __int64 v55; // [rsp+E0h] [rbp-330h]
  __int64 v56; // [rsp+E8h] [rbp-328h]
  __int64 v57; // [rsp+F0h] [rbp-320h]
  _QWORD *v58; // [rsp+F8h] [rbp-318h]
  char v59; // [rsp+100h] [rbp-310h]
  __int64 v60; // [rsp+110h] [rbp-300h] BYREF
  _BYTE *v61; // [rsp+118h] [rbp-2F8h]
  _BYTE *v62; // [rsp+120h] [rbp-2F0h]
  __int64 v63; // [rsp+128h] [rbp-2E8h]
  int v64; // [rsp+130h] [rbp-2E0h]
  _BYTE v65[136]; // [rsp+138h] [rbp-2D8h] BYREF
  _BYTE *v66; // [rsp+1C0h] [rbp-250h] BYREF
  __int64 v67; // [rsp+1C8h] [rbp-248h]
  _BYTE v68[256]; // [rsp+1D0h] [rbp-240h] BYREF
  _BYTE *v69; // [rsp+2D0h] [rbp-140h] BYREF
  __int64 v70; // [rsp+2D8h] [rbp-138h]
  _BYTE v71[304]; // [rsp+2E0h] [rbp-130h] BYREF

  v4 = *(unsigned __int8 *)(a1 + 96);
  if ( (_BYTE)v4 )
    return 1;
  v6 = *(_QWORD *)(a4 + 40);
  v46 = *(_QWORD *)(a4 + 32);
  v10 = sub_1D5A730(a1, v46, *(_QWORD *)(a3 + 32), *(_QWORD *)(a3 + 40));
  v11 = *(_QWORD *)(a3 + 40);
  v12 = *(_QWORD *)(a3 + 32);
  if ( v10 )
  {
    v39 = sub_1D5A730(a1, v6, v12, v11);
    v14 = v6;
    if ( v39 )
      return 1;
  }
  else
  {
    v13 = sub_1D5A730(a1, v6, v12, v11);
    v14 = v46;
    v15 = v46 | v6;
    if ( !v13 )
      v14 = v15;
  }
  if ( !v14 )
    return 1;
  v16 = *(_QWORD *)(a1 + 8);
  v17 = *(_QWORD *)(a1 + 16);
  v60 = 0;
  v67 = 0x1000000000LL;
  v66 = v68;
  v61 = v65;
  v62 = v65;
  v63 = 16;
  v64 = 0;
  if ( (unsigned __int8)sub_1D5CA40(a2, (__int64)&v66, (__int64)&v60, v16, v17, 0) )
    goto LABEL_30;
  v70 = 0x2000000000LL;
  v69 = v71;
  if ( !(_DWORD)v67 )
  {
    v4 = 1;
    goto LABEL_30;
  }
  v18 = 0;
  v40 = 16LL * (unsigned int)v67;
  while ( 1 )
  {
    v19 = *(_QWORD *)&v66[v18];
    v20 = (*(_BYTE *)(v19 + 23) & 0x40) != 0 ? *(_QWORD *)(v19 - 8) : v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF);
    v21 = *(_QWORD *)(v20 + 24LL * *(unsigned int *)&v66[v18 + 8]);
    v22 = *(_QWORD *)v21;
    if ( *(_BYTE *)(*(_QWORD *)v21 + 8LL) != 15 )
      break;
    v23 = *(_QWORD *)(v22 + 24);
    v24 = *(_DWORD *)(v22 + 8);
    v25 = 0;
    v26 = *(_QWORD *)(a1 + 80);
    memset(v49, 0, sizeof(v49));
    v47 = v24 >> 8;
    v50 = 0;
    v48[1] = 0;
    v27 = *(unsigned int *)(v26 + 8);
    v48[0] = 0;
    if ( (_DWORD)v27 )
      v25 = *(_QWORD *)(*(_QWORD *)v26 + 8 * v27 - 8);
    v28 = *(_QWORD *)(a1 + 48);
    v41 = v23;
    v29 = *(_QWORD *)(a1 + 64);
    v42 = v26;
    v43 = *(_QWORD *)(a1 + 72);
    v30 = *(_QWORD *)(a1 + 16);
    v51[1] = *(_QWORD *)(a1 + 8);
    v44 = v29;
    v51[0] = &v69;
    v45 = v28;
    v51[2] = v30;
    v31 = sub_15F2050(v28);
    v51[3] = sub_1632FA0(v31);
    v56 = v43;
    v52 = v47;
    v54 = v49;
    v57 = v42;
    v53 = v45;
    v51[4] = v41;
    v55 = v44;
    v58 = v48;
    v59 = 1;
    sub_1D61F00((__int64)v51, v21, 0);
    sub_1D5ABA0(*(__int64 **)(a1 + 80), v25);
    v32 = v69;
    v33 = 8LL * (unsigned int)v70;
    v34 = &v69[v33];
    v35 = v33 >> 3;
    v36 = v33 >> 5;
    if ( v36 )
    {
      v37 = v69;
      v38 = &v69[32 * v36];
      while ( a2 != *v37 )
      {
        if ( a2 == v37[1] )
        {
          ++v37;
          goto LABEL_23;
        }
        if ( a2 == v37[2] )
        {
          v37 += 2;
          goto LABEL_23;
        }
        if ( a2 == v37[3] )
        {
          v37 += 3;
          goto LABEL_23;
        }
        v37 += 4;
        if ( v38 == v37 )
        {
          v35 = v34 - v37;
          goto LABEL_38;
        }
      }
      goto LABEL_23;
    }
    v37 = v69;
LABEL_38:
    if ( v35 == 2 )
      goto LABEL_42;
    if ( v35 == 3 )
    {
      if ( a2 == *v37 )
        goto LABEL_23;
      ++v37;
LABEL_42:
      if ( a2 == *v37 )
        goto LABEL_23;
      ++v37;
      goto LABEL_44;
    }
    if ( v35 != 1 )
      goto LABEL_27;
LABEL_44:
    if ( a2 != *v37 )
      goto LABEL_27;
LABEL_23:
    if ( v34 == v37 )
      goto LABEL_27;
    LODWORD(v70) = 0;
    v18 += 16;
    if ( v40 == v18 )
    {
      v4 = 1;
      goto LABEL_28;
    }
  }
  v32 = v69;
LABEL_27:
  v4 = 0;
LABEL_28:
  if ( v32 != v71 )
    _libc_free((unsigned __int64)v32);
LABEL_30:
  if ( v62 != v61 )
    _libc_free((unsigned __int64)v62);
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  return v4;
}
