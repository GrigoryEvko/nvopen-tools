// Function: sub_1C0CC70
// Address: 0x1c0cc70
//
__int64 __fastcall sub_1C0CC70(_QWORD *a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  bool v9; // zf
  __int64 v10; // r11
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 v15; // r9
  int v16; // r13d
  __int64 *v17; // r14
  unsigned int v18; // ecx
  __int64 *v19; // r8
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rdi
  __int64 v23; // r8
  __int64 v24; // rdx
  __int64 *v25; // rax
  int v26; // edx
  __int64 v27; // r14
  unsigned int v28; // esi
  __int64 *v29; // rcx
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // r13
  char v33; // al
  int v34; // ecx
  __int64 v35; // rdx
  __int64 v36; // rbx
  __int64 v37; // r11
  __int64 v38; // r15
  __int64 v39; // r8
  __int64 *v40; // rdi
  __int64 v41; // rcx
  unsigned int v42; // esi
  __int64 *v43; // r14
  __int64 *v44; // r10
  int v45; // edx
  int v46; // eax
  __int64 v47; // rbx
  __int64 v48; // r12
  unsigned int v49; // esi
  int v50; // r9d
  __int64 v51; // r8
  int v52; // r11d
  __int64 *v53; // r10
  unsigned int v54; // edx
  __int64 *v55; // rdi
  __int64 v56; // rcx
  int v57; // ecx
  int v58; // ecx
  __int64 v59; // rbx
  __int64 v60; // rax
  int v61; // edi
  __int64 v62; // rdi
  char v63; // al
  int v64; // r10d
  __int64 v65; // [rsp+18h] [rbp-B8h]
  int v66; // [rsp+18h] [rbp-B8h]
  __int64 v67; // [rsp+20h] [rbp-B0h]
  __int64 v68; // [rsp+28h] [rbp-A8h]
  __int64 v70; // [rsp+38h] [rbp-98h]
  __int64 v72; // [rsp+48h] [rbp-88h]
  __int64 v73; // [rsp+48h] [rbp-88h]
  __int64 v76; // [rsp+60h] [rbp-70h] BYREF
  __int64 v77; // [rsp+68h] [rbp-68h] BYREF
  _QWORD v78[12]; // [rsp+70h] [rbp-60h] BYREF

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result <= 0x17u )
  {
    if ( (_BYTE)result == 17 )
    {
      result = sub_1C2F070(*a1);
      if ( !(_BYTE)result )
        goto LABEL_23;
    }
    return result;
  }
  if ( (unsigned __int8)(result - 58) <= 1u )
    goto LABEL_3;
  if ( (_BYTE)result == 78 )
  {
    v62 = *(_QWORD *)(a2 - 24);
    v63 = *(_BYTE *)(v62 + 16);
    if ( v63 == 20 )
    {
      result = sub_1C090D0(v62);
    }
    else
    {
      if ( v63 || (*(_BYTE *)(v62 + 33) & 0x20) == 0 )
        goto LABEL_90;
      result = sub_1C07700(*(_DWORD *)(v62 + 36));
    }
    if ( (_DWORD)result )
      goto LABEL_4;
LABEL_90:
    if ( (unsigned __int8)sub_1C07900(a2) )
    {
LABEL_91:
      result = 0;
      goto LABEL_4;
    }
    goto LABEL_3;
  }
  if ( (_BYTE)result != 54 )
    goto LABEL_91;
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
LABEL_3:
    result = 7;
    goto LABEL_4;
  }
  result = 0;
LABEL_4:
  v9 = ((unsigned int)result | *a3) == 0;
  *a3 |= result;
  if ( !v9 )
    return result;
  result = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( *(_BYTE *)(a2 + 16) != 77 )
  {
    if ( !(_DWORD)result )
      return result;
    v47 = 0;
    v48 = 24 * result;
    while ( 1 )
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v49 = *(_DWORD *)(a4 + 24);
        result = *(_QWORD *)(*(_QWORD *)(a2 - 8) + v47);
        v77 = result;
        if ( !v49 )
          goto LABEL_55;
      }
      else
      {
        v49 = *(_DWORD *)(a4 + 24);
        result = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) + v47);
        v77 = result;
        if ( !v49 )
        {
LABEL_55:
          ++*(_QWORD *)a4;
          goto LABEL_56;
        }
      }
      v50 = v49 - 1;
      v51 = *(_QWORD *)(a4 + 8);
      v52 = 1;
      v53 = 0;
      v54 = (v49 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
      v55 = (__int64 *)(v51 + 8LL * v54);
      v56 = *v55;
      if ( result != *v55 )
        break;
LABEL_52:
      v47 += 24;
      if ( v48 == v47 )
        return result;
    }
    while ( v56 != -8 )
    {
      if ( v53 || v56 != -16 )
        v55 = v53;
      v54 = v50 & (v52 + v54);
      v56 = *(_QWORD *)(v51 + 8LL * v54);
      if ( result == v56 )
        goto LABEL_52;
      ++v52;
      v53 = v55;
      v55 = (__int64 *)(v51 + 8LL * v54);
    }
    if ( !v53 )
      v53 = v55;
    v61 = *(_DWORD *)(a4 + 16);
    ++*(_QWORD *)a4;
    v57 = v61 + 1;
    if ( 4 * (v61 + 1) >= 3 * v49 )
    {
LABEL_56:
      v49 *= 2;
    }
    else if ( v49 - *(_DWORD *)(a4 + 20) - v57 > v49 >> 3 )
    {
      goto LABEL_81;
    }
    sub_1353F00(a4, v49);
    sub_1A97120(a4, &v77, v78);
    v53 = (__int64 *)v78[0];
    result = v77;
    v57 = *(_DWORD *)(a4 + 16) + 1;
LABEL_81:
    *(_DWORD *)(a4 + 16) = v57;
    if ( *v53 != -8 )
      --*(_DWORD *)(a4 + 20);
    *v53 = result;
    result = *(unsigned int *)(a5 + 8);
    if ( (unsigned int)result >= *(_DWORD *)(a5 + 12) )
    {
      sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v51, v50);
      result = *(unsigned int *)(a5 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a5 + 8 * result) = v77;
    ++*(_DWORD *)(a5 + 8);
    goto LABEL_52;
  }
  v68 = *(_QWORD *)(a2 + 40);
  if ( !(_DWORD)result )
    return result;
  v10 = a2;
  v70 = 8 * result;
  v11 = 0;
  while ( 1 )
  {
    v33 = *(_BYTE *)(v10 + 23) & 0x40;
    if ( v33 )
      v12 = *(_QWORD *)(v10 - 8);
    else
      v12 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
    v13 = *(_QWORD *)(v12 + 3 * v11);
    v14 = *(_DWORD *)(a4 + 24);
    v76 = v13;
    if ( !v14 )
    {
      ++*(_QWORD *)a4;
LABEL_93:
      v67 = v10;
      v59 = a4;
      v14 *= 2;
LABEL_94:
      sub_1353F00(v59, v14);
      sub_1A97120(v59, &v76, v78);
      v17 = (__int64 *)v78[0];
      v13 = v76;
      v10 = v67;
      v58 = *(_DWORD *)(v59 + 16) + 1;
      goto LABEL_67;
    }
    v15 = *(_QWORD *)(a4 + 8);
    v16 = 1;
    v17 = 0;
    v18 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v19 = (__int64 *)(v15 + 8LL * v18);
    v20 = *v19;
    if ( v13 == *v19 )
      goto LABEL_11;
    while ( v20 != -8 )
    {
      if ( v17 || v20 != -16 )
        v19 = v17;
      v18 = (v14 - 1) & (v16 + v18);
      v20 = *(_QWORD *)(v15 + 8LL * v18);
      if ( v13 == v20 )
        goto LABEL_11;
      ++v16;
      v17 = v19;
      v19 = (__int64 *)(v15 + 8LL * v18);
    }
    if ( !v17 )
      v17 = v19;
    ++*(_QWORD *)a4;
    v58 = *(_DWORD *)(a4 + 16) + 1;
    if ( 4 * v58 >= 3 * v14 )
      goto LABEL_93;
    v59 = a4;
    if ( v14 - *(_DWORD *)(a4 + 20) - v58 <= v14 >> 3 )
    {
      v67 = v10;
      goto LABEL_94;
    }
LABEL_67:
    *(_DWORD *)(a4 + 16) = v58;
    if ( *v17 != -8 )
      --*(_DWORD *)(a4 + 20);
    *v17 = v13;
    v60 = *(unsigned int *)(a5 + 8);
    if ( (unsigned int)v60 >= *(_DWORD *)(a5 + 12) )
    {
      v73 = v10;
      sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, (int)v19, v15);
      v60 = *(unsigned int *)(a5 + 8);
      v10 = v73;
    }
    *(_QWORD *)(*(_QWORD *)a5 + 8 * v60) = v76;
    ++*(_DWORD *)(a5 + 8);
    v33 = *(_BYTE *)(v10 + 23) & 0x40;
LABEL_11:
    if ( v33 )
      v21 = *(_QWORD *)(v10 - 8);
    else
      v21 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
    v22 = a1[13];
    v23 = *(_QWORD *)(v22 + 48);
    v24 = *(unsigned int *)(v22 + 64);
    v25 = (__int64 *)(v23 + 16 * v24);
    if ( !(_DWORD)v24 )
      goto LABEL_23;
    v26 = v24 - 1;
    v27 = *(_QWORD *)(v11 + v21 + 24LL * *(unsigned int *)(v10 + 56) + 8);
    v28 = v26 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
    v29 = (__int64 *)(v23 + 16LL * v28);
    v30 = *v29;
    if ( v27 != *v29 )
      break;
LABEL_15:
    if ( v25 == v29 )
      goto LABEL_23;
    v72 = v10;
    v31 = sub_1C0A150(v22, v27);
    v10 = v72;
    v32 = v31;
    result = *(unsigned int *)(v31 + 56);
    if ( (_DWORD)result )
    {
      v77 = sub_1C0A960((int *)a1[13], v27, v68);
      result = sub_1C0CB20((__int64)v78, a6, &v77);
      v10 = v72;
      goto LABEL_18;
    }
    v35 = *(unsigned int *)(v32 + 32);
    v36 = 0;
    if ( (_DWORD)v35 )
    {
      v37 = v11;
      v38 = 8 * v35;
      while ( 1 )
      {
        v42 = *(_DWORD *)(a6 + 24);
        v43 = (__int64 *)(v36 + *(_QWORD *)(v32 + 24));
        if ( !v42 )
          break;
        v39 = *(_QWORD *)(a6 + 8);
        result = (v42 - 1) & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
        v40 = (__int64 *)(v39 + 8 * result);
        v41 = *v40;
        if ( *v43 == *v40 )
        {
LABEL_29:
          v36 += 8;
          if ( v36 == v38 )
            goto LABEL_37;
        }
        else
        {
          v66 = 1;
          v44 = 0;
          while ( v41 != -8 )
          {
            if ( v41 != -16 || v44 )
              v40 = v44;
            result = (v42 - 1) & (v66 + (_DWORD)result);
            v41 = *(_QWORD *)(v39 + 8LL * (unsigned int)result);
            if ( *v43 == v41 )
              goto LABEL_29;
            ++v66;
            v44 = v40;
            v40 = (__int64 *)(v39 + 8LL * (unsigned int)result);
          }
          v46 = *(_DWORD *)(a6 + 16);
          if ( !v44 )
            v44 = v40;
          ++*(_QWORD *)a6;
          v45 = v46 + 1;
          if ( 4 * (v46 + 1) >= 3 * v42 )
            goto LABEL_32;
          if ( v42 - *(_DWORD *)(a6 + 20) - v45 > v42 >> 3 )
            goto LABEL_34;
          v65 = v37;
LABEL_33:
          sub_1C0C970(a6, v42);
          sub_1C09C20(a6, v43, v78);
          v44 = (__int64 *)v78[0];
          v37 = v65;
          v45 = *(_DWORD *)(a6 + 16) + 1;
LABEL_34:
          *(_DWORD *)(a6 + 16) = v45;
          if ( *v44 != -8 )
            --*(_DWORD *)(a6 + 20);
          result = *v43;
          v36 += 8;
          *v44 = *v43;
          if ( v36 == v38 )
          {
LABEL_37:
            v11 = v37;
            v10 = v72;
            goto LABEL_18;
          }
        }
      }
      ++*(_QWORD *)a6;
LABEL_32:
      v65 = v37;
      v42 *= 2;
      goto LABEL_33;
    }
LABEL_18:
    v11 += 8;
    if ( v70 == v11 )
      return result;
  }
  v34 = 1;
  while ( v30 != -8 )
  {
    v64 = v34 + 1;
    v28 = v26 & (v34 + v28);
    v29 = (__int64 *)(v23 + 16LL * v28);
    v30 = *v29;
    if ( v27 == *v29 )
      goto LABEL_15;
    v34 = v64;
  }
LABEL_23:
  result = (__int64)a3;
  *a3 = 7;
  return result;
}
