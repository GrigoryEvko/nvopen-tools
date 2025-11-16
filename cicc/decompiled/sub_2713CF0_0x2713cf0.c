// Function: sub_2713CF0
// Address: 0x2713cf0
//
__int64 __fastcall sub_2713CF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _BYTE *v4; // r13
  __int64 v6; // rbx
  unsigned int v7; // r14d
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  unsigned int v12; // edx
  __int64 *v13; // r15
  __int64 v14; // r8
  unsigned int v15; // esi
  __int64 v16; // r10
  __int64 v17; // r8
  _QWORD *v18; // rdx
  int v19; // r11d
  unsigned int v20; // edi
  _QWORD *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rbx
  unsigned int v24; // ecx
  char v25; // dl
  int v26; // eax
  int v27; // eax
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  int v33; // r10d
  int v34; // ecx
  int v35; // ecx
  __int64 v36; // r8
  unsigned int v37; // edi
  __int64 v38; // rsi
  int v39; // r11d
  _QWORD *v40; // r10
  char v41; // bl
  char v42; // bl
  char v43; // cl
  char v44; // al
  int v45; // ecx
  int v46; // ecx
  __int64 v47; // r8
  int v48; // r11d
  __int64 v49; // rsi
  __int64 v50; // rdi
  _BYTE *v52; // [rsp+10h] [rbp-180h]
  _BYTE *v53; // [rsp+18h] [rbp-178h]
  char v54; // [rsp+25h] [rbp-16Bh]
  unsigned __int8 v55; // [rsp+26h] [rbp-16Ah]
  char v56; // [rsp+27h] [rbp-169h]
  unsigned int v57; // [rsp+38h] [rbp-158h]
  int v58; // [rsp+3Ch] [rbp-154h]
  __int64 v59; // [rsp+40h] [rbp-150h]
  _BYTE *v60; // [rsp+48h] [rbp-148h]
  __int64 v61; // [rsp+68h] [rbp-128h] BYREF
  _BYTE *v62; // [rsp+70h] [rbp-120h]
  __int64 v63; // [rsp+78h] [rbp-118h]
  int v64; // [rsp+80h] [rbp-110h]
  char v65; // [rsp+84h] [rbp-10Ch]
  _BYTE v66[16]; // [rsp+88h] [rbp-108h] BYREF
  __int64 v67; // [rsp+98h] [rbp-F8h] BYREF
  _BYTE *v68; // [rsp+A0h] [rbp-F0h]
  __int64 v69; // [rsp+A8h] [rbp-E8h]
  int v70; // [rsp+B0h] [rbp-E0h]
  char v71; // [rsp+B4h] [rbp-DCh]
  _BYTE v72[16]; // [rsp+B8h] [rbp-D8h] BYREF
  char v73; // [rsp+C8h] [rbp-C8h]
  __int64 v74; // [rsp+D0h] [rbp-C0h] BYREF
  __int16 v75; // [rsp+D8h] [rbp-B8h]
  char v76; // [rsp+DAh] [rbp-B6h]
  __int64 v77; // [rsp+E0h] [rbp-B0h]
  __int64 v78; // [rsp+E8h] [rbp-A8h]
  char v79[8]; // [rsp+F0h] [rbp-A0h] BYREF
  unsigned __int64 v80; // [rsp+F8h] [rbp-98h]
  char v81; // [rsp+10Ch] [rbp-84h]
  char v82[16]; // [rsp+110h] [rbp-80h] BYREF
  char v83[8]; // [rsp+120h] [rbp-70h] BYREF
  unsigned __int64 v84; // [rsp+128h] [rbp-68h]
  char v85; // [rsp+13Ch] [rbp-54h]
  _BYTE v86[80]; // [rsp+140h] [rbp-50h] BYREF

  result = *(_QWORD *)(a3 + 48);
  v4 = *(_BYTE **)(a3 + 40);
  v53 = (_BYTE *)result;
  if ( v4 != (_BYTE *)result )
  {
LABEL_4:
    if ( !v4[10] )
      goto LABEL_3;
    v59 = *(_QWORD *)v4;
    result = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( result == a1 + 48 )
      goto LABEL_3;
    if ( !result )
      BUG();
    v6 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result > 0xA )
      goto LABEL_3;
    result = sub_B46E30(v6);
    v58 = result;
    if ( !(_DWORD)result )
      goto LABEL_3;
    v55 = 0;
    v7 = 0;
    v52 = v4 + 8;
    v54 = 1;
    v56 = 0;
    v60 = v4;
    v8 = v6;
    v57 = ((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4);
    while ( 1 )
    {
      v9 = sub_B46EC0(v8, v7);
      v10 = *(unsigned int *)(a2 + 24);
      v11 = *(_QWORD *)(a2 + 8);
      if ( (_DWORD)v10 )
      {
        v12 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v13 = (__int64 *)(v11 + 192LL * v12);
        v14 = *v13;
        if ( v9 == *v13 )
        {
LABEL_12:
          v15 = *((_DWORD *)v13 + 24);
          v16 = (__int64)(v13 + 9);
          if ( !v15 )
            goto LABEL_57;
          goto LABEL_13;
        }
        v33 = 1;
        while ( v14 != -4096 )
        {
          v12 = (v10 - 1) & (v33 + v12);
          v13 = (__int64 *)(v11 + 192LL * v12);
          v14 = *v13;
          if ( v9 == *v13 )
            goto LABEL_12;
          ++v33;
        }
      }
      v13 = (__int64 *)(v11 + 192 * v10);
      v15 = *((_DWORD *)v13 + 24);
      v16 = (__int64)(v13 + 9);
      if ( !v15 )
      {
LABEL_57:
        ++v13[9];
        goto LABEL_58;
      }
LABEL_13:
      v17 = v13[10];
      v18 = 0;
      v19 = 1;
      v20 = (v15 - 1) & v57;
      v21 = (_QWORD *)(v17 + 16LL * v20);
      v22 = *v21;
      if ( v59 == *v21 )
      {
LABEL_14:
        v23 = v13[13] + 136LL * v21[1] + 8;
        goto LABEL_15;
      }
      while ( v22 != -4096 )
      {
        if ( v22 == -8192 && !v18 )
          v18 = v21;
        v20 = (v15 - 1) & (v19 + v20);
        v21 = (_QWORD *)(v17 + 16LL * v20);
        v22 = *v21;
        if ( v59 == *v21 )
          goto LABEL_14;
        ++v19;
      }
      if ( !v18 )
        v18 = v21;
      v26 = *((_DWORD *)v13 + 22);
      ++v13[9];
      v27 = v26 + 1;
      if ( 4 * v27 < 3 * v15 )
      {
        if ( v15 - *((_DWORD *)v13 + 23) - v27 <= v15 >> 3 )
        {
          sub_2712CF0(v16, v15);
          v45 = *((_DWORD *)v13 + 24);
          if ( !v45 )
          {
LABEL_95:
            ++*((_DWORD *)v13 + 22);
LABEL_96:
            BUG();
          }
          v46 = v45 - 1;
          v47 = v13[10];
          v40 = 0;
          v48 = 1;
          LODWORD(v49) = v46 & v57;
          v27 = *((_DWORD *)v13 + 22) + 1;
          v18 = (_QWORD *)(v47 + 16LL * (v46 & v57));
          v50 = *v18;
          if ( v59 != *v18 )
          {
            while ( v50 != -4096 )
            {
              if ( v50 == -8192 && !v40 )
                v40 = v18;
              v49 = v46 & (unsigned int)(v49 + v48);
              v18 = (_QWORD *)(v47 + 16 * v49);
              v50 = *v18;
              if ( v59 == *v18 )
                goto LABEL_41;
              ++v48;
            }
LABEL_62:
            if ( v40 )
              v18 = v40;
            goto LABEL_41;
          }
        }
        goto LABEL_41;
      }
LABEL_58:
      sub_2712CF0(v16, 2 * v15);
      v34 = *((_DWORD *)v13 + 24);
      if ( !v34 )
        goto LABEL_95;
      v35 = v34 - 1;
      v36 = v13[10];
      v37 = v35 & v57;
      v27 = *((_DWORD *)v13 + 22) + 1;
      v18 = (_QWORD *)(v36 + 16LL * (v35 & v57));
      v38 = *v18;
      if ( v59 != *v18 )
      {
        v39 = 1;
        v40 = 0;
        while ( v38 != -4096 )
        {
          if ( !v40 && v38 == -8192 )
            v40 = v18;
          v37 = v35 & (v39 + v37);
          v18 = (_QWORD *)(v36 + 16LL * v37);
          v38 = *v18;
          if ( v59 == *v18 )
            goto LABEL_41;
          ++v39;
        }
        goto LABEL_62;
      }
LABEL_41:
      *((_DWORD *)v13 + 22) = v27;
      if ( *v18 != -4096 )
        --*((_DWORD *)v13 + 23);
      v18[1] = 0;
      *v18 = v59;
      v28 = v13[14] - v13[13];
      v74 = v59;
      v18[1] = 0xF0F0F0F0F0F0F0F1LL * (v28 >> 3);
      v75 = 0;
      v62 = v66;
      v68 = v72;
      v61 = 0;
      v63 = 2;
      v64 = 0;
      v65 = 1;
      v67 = 0;
      v69 = 2;
      v70 = 0;
      v71 = 1;
      v73 = 0;
      v76 = 0;
      v77 = 0;
      v78 = 0;
      sub_C8CF70((__int64)v79, v82, 2, (__int64)v66, (__int64)&v61);
      sub_C8CF70((__int64)v83, v86, 2, (__int64)v72, (__int64)&v67);
      v86[16] = v73;
      sub_2712480((unsigned __int64 *)v13 + 13, (__int64)&v74, v29, v30, v31, v32);
      if ( !v85 )
        _libc_free(v84);
      if ( v81 )
      {
        if ( !v71 )
          goto LABEL_72;
      }
      else
      {
        _libc_free(v80);
        if ( !v71 )
LABEL_72:
          _libc_free((unsigned __int64)v68);
      }
      if ( !v65 )
        _libc_free((unsigned __int64)v62);
      v23 = v13[13] + v28 + 8;
LABEL_15:
      result = *(unsigned __int8 *)(v23 + 2);
      if ( !(_BYTE)result )
        goto LABEL_30;
      v24 = *(unsigned __int8 *)(v23 + 8);
      v25 = v60[10];
      if ( v25 == 2 )
      {
        if ( (unsigned __int8)result <= 5u )
        {
          if ( (unsigned __int8)result > 2u )
          {
            v42 = v54;
            v43 = v60[16] | v24;
            v44 = v55;
            if ( v43 )
              v44 = v43;
            v55 = v44;
            result = 0;
            if ( !v43 )
              v42 = 0;
            v54 = v42;
            goto LABEL_24;
          }
          if ( (_BYTE)result == 1 )
            goto LABEL_96;
LABEL_53:
          v56 = 1;
        }
      }
      else if ( v25 == 3 )
      {
        if ( (_BYTE)result == 3 )
          goto LABEL_53;
        if ( (unsigned __int8)result > 3u )
        {
          result = (unsigned int)(result - 4);
          if ( (unsigned __int8)result <= 1u )
          {
            v41 = v54;
            LOBYTE(v24) = v60[16] | v24;
            if ( !(_BYTE)v24 )
              v41 = 0;
            result = v55;
            if ( (_BYTE)v24 )
              result = v24;
            v54 = v41;
            v55 = result;
          }
          goto LABEL_24;
        }
        if ( (_BYTE)result == 1 )
          goto LABEL_96;
        if ( !v60[16] && !(_BYTE)v24 )
        {
LABEL_30:
          result = sub_271D520(v52, 0);
          goto LABEL_24;
        }
        result = (__int64)v60;
        v60[128] = 1;
      }
LABEL_24:
      if ( v58 == ++v7 )
      {
        v4 = v60;
        if ( v56 && !v54 )
        {
          result = sub_271D520(v52, 0);
          goto LABEL_3;
        }
        if ( v55 )
        {
          v60[128] = 1;
          v4 = v60 + 136;
          if ( v53 == v60 + 136 )
            return result;
        }
        else
        {
LABEL_3:
          v4 += 136;
          if ( v53 == v4 )
            return result;
        }
        goto LABEL_4;
      }
    }
  }
  return result;
}
