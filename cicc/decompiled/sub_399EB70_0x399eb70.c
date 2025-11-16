// Function: sub_399EB70
// Address: 0x399eb70
//
__int64 __fastcall sub_399EB70(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 result; // rax
  int v5; // r8d
  int v6; // r9d
  _QWORD *v7; // rbx
  int v8; // eax
  int v9; // r8d
  int v10; // r9d
  __int16 *v11; // rdx
  __int16 v12; // ax
  __int16 *v13; // rdx
  unsigned __int16 v14; // r15
  int v15; // eax
  __int16 v16; // ax
  __int16 *v17; // r13
  _WORD *v19; // rdx
  __int16 *v20; // r15
  unsigned int v21; // r12d
  unsigned int v22; // r13d
  unsigned __int64 v23; // r8
  unsigned __int64 v24; // r9
  unsigned int v25; // r14d
  unsigned __int64 *v26; // rax
  unsigned __int64 v27; // r12
  char v28; // r11
  char v29; // si
  unsigned int v30; // esi
  unsigned int v31; // ecx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // r12
  unsigned __int64 v36; // r12
  unsigned __int64 *v37; // rdx
  _QWORD *v38; // rax
  __int16 v39; // ax
  unsigned __int64 v40; // r12
  __int64 v41; // rdx
  _QWORD *v42; // rdx
  unsigned __int64 *v43; // rsi
  __int64 v44; // r10
  unsigned __int64 v45; // r8
  unsigned int v46; // ecx
  unsigned int i; // edx
  unsigned __int64 v48; // rax
  unsigned int v49; // ecx
  __int64 v50; // rax
  unsigned __int64 v51; // rax
  unsigned int v52; // eax
  unsigned int v53; // ecx
  __int64 v54; // rdx
  char v55; // r10
  __int64 v56; // rdi
  unsigned __int64 *v57; // rsi
  unsigned __int64 v58; // r9
  unsigned int v59; // ecx
  unsigned int j; // edx
  unsigned __int64 *v61; // rdx
  __int64 v62; // rbx
  __int64 v63; // rdx
  _QWORD *v64; // rdx
  __int64 v65; // rbx
  __int64 v66; // rdx
  _QWORD *v67; // rdx
  __int64 v68; // r14
  unsigned int v69; // r12d
  int v70; // r13d
  int v71; // r8d
  int v72; // r9d
  int v73; // r12d
  __int64 v74; // rax
  _QWORD *v75; // rax
  unsigned __int64 v76; // r12
  __int64 v77; // rdx
  unsigned __int64 *v78; // rdx
  unsigned int v79; // [rsp+Ch] [rbp-74h]
  unsigned __int64 v80; // [rsp+10h] [rbp-70h]
  unsigned int v81; // [rsp+18h] [rbp-68h]
  int v82; // [rsp+1Ch] [rbp-64h]
  unsigned int v83; // [rsp+20h] [rbp-60h]
  unsigned __int16 v84; // [rsp+28h] [rbp-58h]
  unsigned __int8 v87; // [rsp+34h] [rbp-4Ch]
  unsigned __int8 v88; // [rsp+34h] [rbp-4Ch]
  unsigned __int8 v90; // [rsp+38h] [rbp-48h]
  unsigned __int64 v91; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v92; // [rsp+48h] [rbp-38h] BYREF

  if ( (int)a3 <= 0 )
  {
    result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL))(a1);
    if ( (_BYTE)result )
    {
      v41 = *(unsigned int *)(a1 + 16);
      if ( (unsigned int)v41 >= *(_DWORD *)(a1 + 20) )
      {
        v88 = result;
        sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 16, v5, v6);
        v41 = *(unsigned int *)(a1 + 16);
        result = v88;
      }
      v42 = (_QWORD *)(*(_QWORD *)(a1 + 8) + 16 * v41);
      *v42 = 0xFFFFFFFFLL;
      v42[1] = 0;
      ++*(_DWORD *)(a1 + 16);
    }
    else
    {
      result = *(unsigned __int8 *)(a1 + 80);
      if ( (_BYTE)result )
      {
        v65 = a3;
        v66 = *(unsigned int *)(a1 + 16);
        if ( (unsigned int)v66 >= *(_DWORD *)(a1 + 20) )
        {
          v87 = *(_BYTE *)(a1 + 80);
          sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 16, v5, v6);
          v66 = *(unsigned int *)(a1 + 16);
          result = v87;
        }
        v67 = (_QWORD *)(*(_QWORD *)(a1 + 8) + 16 * v66);
        *v67 = v65;
        v67[1] = 0;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return result;
  }
  v7 = (_QWORD *)(a2 + 8);
  v8 = sub_38D70E0(a2 + 8, a3, 0);
  if ( v8 >= 0 )
  {
    v62 = (unsigned int)v8;
    v63 = *(unsigned int *)(a1 + 16);
    if ( (unsigned int)v63 >= *(_DWORD *)(a1 + 20) )
    {
      sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 16, v9, v10);
      v63 = *(unsigned int *)(a1 + 16);
    }
    v64 = (_QWORD *)(*(_QWORD *)(a1 + 8) + 16 * v63);
    *v64 = v62;
    v64[1] = 0;
    ++*(_DWORD *)(a1 + 16);
    return 1;
  }
  v11 = (__int16 *)(*(_QWORD *)(a2 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(a2 + 8) + 24LL * a3 + 8));
  v12 = *v11;
  v13 = v11 + 1;
  v14 = v12 + a3;
  if ( !v12 )
    v13 = 0;
LABEL_11:
  v17 = v13;
  while ( v17 )
  {
    v15 = sub_38D70E0((__int64)v7, v14, 0);
    if ( v15 >= 0 )
    {
      v68 = (unsigned int)v15;
      v69 = sub_38D7050(v7, v14, a3);
      v70 = sub_38D70C0((__int64)v7, v69);
      v73 = sub_38D70D0((__int64)v7, v69);
      v74 = *(unsigned int *)(a1 + 16);
      if ( (unsigned int)v74 >= *(_DWORD *)(a1 + 20) )
      {
        sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 16, v71, v72);
        v74 = *(unsigned int *)(a1 + 16);
      }
      v75 = (_QWORD *)(*(_QWORD *)(a1 + 8) + 16 * v74);
      *v75 = v68;
      v75[1] = "super-register";
      ++*(_DWORD *)(a1 + 16);
      *(_DWORD *)(a1 + 68) = v70;
      *(_DWORD *)(a1 + 72) = v73;
      return 1;
    }
    v16 = *v17;
    v13 = 0;
    ++v17;
    v14 += v16;
    if ( !v16 )
      goto LABEL_11;
  }
  v79 = *(_DWORD *)(*(_QWORD *)(a2 + 280)
                  + 24LL
                  * (*(unsigned __int16 *)(*sub_1F4ABE0(a2, a3, 1) + 24)
                   + *(_DWORD *)(a2 + 288)
                   * (unsigned int)((__int64)(*(_QWORD *)(a2 + 264) - *(_QWORD *)(a2 + 256)) >> 3)));
  sub_13A4F10(&v91, v79, 0);
  v19 = (_WORD *)(*(_QWORD *)(a2 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(a2 + 8) + 24LL * a3 + 4));
  result = 0;
  if ( !*v19 )
    goto LABEL_42;
  v81 = 0;
  v20 = v19 + 1;
  v84 = a3 + *v19;
  while ( 1 )
  {
    v21 = sub_38D7050(v7, a3, v84);
    v83 = sub_38D70C0((__int64)v7, v21);
    v22 = sub_38D70D0((__int64)v7, v21);
    v82 = sub_38D70E0((__int64)v7, v84, 0);
    if ( v82 < 0 )
      goto LABEL_39;
    sub_13A4F10(&v92, v79, 0);
    v25 = v83 + v22;
    if ( v83 + v22 == v22 )
      goto LABEL_56;
    v26 = (unsigned __int64 *)v92;
    if ( (v92 & 1) != 0 )
    {
      v23 = 1LL << v22;
      v27 = 2
          * ((v92 >> 58 << 57)
           | ~(-1LL << (v92 >> 58)) & (~(-1LL << (v92 >> 58)) & (v92 >> 1) | ((1LL << v25) - (1LL << v22))))
          + 1;
      v92 = v27;
      goto LABEL_19;
    }
    LODWORD(v24) = v22 & 0x3F;
    v43 = (unsigned __int64 *)(*(_QWORD *)v92 + 8LL * (v22 >> 6));
    v44 = 1LL << v25;
    v45 = *v43;
    if ( v22 >> 6 != v25 >> 6 )
    {
      v23 = (-1LL << v24) | v45;
      *v43 = v23;
      v46 = (v22 + 63) & 0xFFFFFFC0;
      for ( i = v46 + 64; v25 >= i; i += 64 )
      {
        *(_QWORD *)(*v26 + 8LL * ((i - 64) >> 6)) = -1;
        v46 = i;
      }
      if ( v25 > v46 )
        *(_QWORD *)(*v26 + 8LL * (v46 >> 6)) |= v44 - 1;
LABEL_56:
      v27 = v92;
      goto LABEL_19;
    }
    v23 = (v44 - (1LL << v24)) | v45;
    *v43 = v23;
    v27 = v92;
LABEL_19:
    v28 = v91 & 1;
    v29 = v27 & 1;
    if ( (v27 & 1) != 0 )
    {
      if ( v28 )
      {
        if ( (~(-1LL << (v27 >> 58)) & (v27 >> 1) & ((-1LL << (v91 >> 58)) | ~(v91 >> 1))) == 0 )
          goto LABEL_39;
        goto LABEL_27;
      }
      v48 = *(unsigned int *)(v91 + 16);
      v80 = v27 >> 58;
    }
    else
    {
      if ( !v28 )
      {
        v30 = (unsigned int)(*(_DWORD *)(v27 + 16) + 63) >> 6;
        v31 = (unsigned int)(*(_DWORD *)(v91 + 16) + 63) >> 6;
        if ( v31 > v30 )
          v31 = (unsigned int)(*(_DWORD *)(v27 + 16) + 63) >> 6;
        if ( v31 )
        {
          v23 = *(_QWORD *)v27;
          LODWORD(v24) = v31;
          v32 = 0;
          while ( (*(_QWORD *)(v23 + 8 * v32) & ~*(_QWORD *)(*(_QWORD *)v91 + 8 * v32)) == 0 )
          {
            if ( v31 == ++v32 )
              goto LABEL_98;
          }
        }
        else
        {
LABEL_98:
          if ( v30 == v31 )
          {
LABEL_37:
            if ( v27 )
            {
              _libc_free(*(_QWORD *)v27);
              j_j___libc_free_0(v27);
            }
            goto LABEL_39;
          }
          while ( !*(_QWORD *)(*(_QWORD *)v27 + 8LL * v31) )
          {
            if ( v30 == ++v31 )
              goto LABEL_37;
          }
        }
        goto LABEL_27;
      }
      v48 = v91 >> 58;
      v80 = *(unsigned int *)(v27 + 16);
    }
    if ( v80 <= v48 )
      v48 = v80;
    LODWORD(v23) = v48;
    if ( v48 )
    {
      v24 = (v27 >> 1) & ~(-1LL << (v27 >> 58));
      v49 = 0;
      while ( 1 )
      {
        v50 = v29 ? (v24 >> v49) & 1 : (*(_QWORD *)(*(_QWORD *)v27 + 8LL * (v49 >> 6)) >> v49) & 1LL;
        if ( (_BYTE)v50 )
        {
          v51 = v28
              ? (((v91 >> 1) & ~(-1LL << (v91 >> 58))) >> v49) & 1
              : (*(_QWORD *)(*(_QWORD *)v91 + 8LL * (v49 >> 6)) >> v49) & 1LL;
          if ( !(_BYTE)v51 )
            break;
        }
        if ( (_DWORD)v23 == ++v49 )
        {
          v52 = v49;
          goto LABEL_72;
        }
      }
    }
    else
    {
      v52 = 0;
LABEL_72:
      if ( (_DWORD)v80 == v52 )
        goto LABEL_36;
      v53 = v52;
      v23 = (v27 >> 1) & ~(-1LL << (v27 >> 58));
      while ( 1 )
      {
        v54 = v29 ? (v23 >> v53) & 1 : (*(_QWORD *)(*(_QWORD *)v27 + 8LL * (v53 >> 6)) >> v53) & 1LL;
        if ( (_BYTE)v54 )
          break;
        if ( (_DWORD)v80 == ++v53 )
          goto LABEL_36;
      }
    }
LABEL_27:
    v33 = *(unsigned int *)(a1 + 16);
    v34 = a1 + 8;
    if ( v22 > v81 )
    {
      if ( (unsigned int)v33 >= *(_DWORD *)(a1 + 20) )
      {
        sub_16CD150(v34, (const void *)(a1 + 24), 0, 16, v23, v24);
        v34 = a1 + 8;
        v33 = *(unsigned int *)(a1 + 16);
      }
      v61 = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + 16 * v33);
      *v61 = ((unsigned __int64)(v22 - v81) << 32) | 0xFFFFFFFF;
      v61[1] = (unsigned __int64)"no DWARF register encoding";
      v33 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
      *(_DWORD *)(a1 + 16) = v33;
    }
    v35 = a4 - v22;
    if ( (unsigned int)v35 > v83 )
      v35 = v83;
    v36 = (unsigned int)v82 | (unsigned __int64)(v35 << 32);
    if ( *(_DWORD *)(a1 + 20) <= (unsigned int)v33 )
    {
      sub_16CD150(v34, (const void *)(a1 + 24), 0, 16, v23, v24);
      v33 = *(unsigned int *)(a1 + 16);
    }
    v37 = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + 16 * v33);
    *v37 = v36;
    v37[1] = (unsigned __int64)"sub-register";
    ++*(_DWORD *)(a1 + 16);
    if ( v22 >= a4 )
      break;
    if ( v25 == v22 )
      goto LABEL_85;
    v38 = (_QWORD *)v91;
    if ( (v91 & 1) != 0 )
    {
      v27 = v92;
      v81 = v83 + v22;
      v23 = 1LL << v22;
      v29 = v92 & 1;
      v91 = 2
          * ((v91 >> 58 << 57)
           | ~(-1LL << (v91 >> 58)) & (((1LL << v25) - (1LL << v22)) | ~(-1LL << (v91 >> 58)) & (v91 >> 1)))
          + 1;
      goto LABEL_36;
    }
    LODWORD(v23) = 1;
    v55 = v22 & 0x3F;
    v56 = 1LL << v25;
    v57 = (unsigned __int64 *)(*(_QWORD *)v91 + 8LL * (v22 >> 6));
    v58 = *v57;
    if ( v22 >> 6 != v25 >> 6 )
    {
      v24 = (-1LL << v55) | v58;
      *v57 = v24;
      v59 = (v22 + 63) & 0xFFFFFFC0;
      for ( j = v59 + 64; v25 >= j; j += 64 )
      {
        *(_QWORD *)(*v38 + 8LL * ((j - 64) >> 6)) = -1;
        v59 = j;
      }
      if ( v25 > v59 )
        *(_QWORD *)(*v38 + 8LL * (v59 >> 6)) |= v56 - 1;
LABEL_85:
      v27 = v92;
      v81 = v83 + v22;
      v29 = v92 & 1;
      goto LABEL_36;
    }
    v81 = v83 + v22;
    v23 = 1LL << v55;
    v24 = (v56 - (1LL << v55)) | v58;
    *v57 = v24;
    v27 = v92;
    v29 = v92 & 1;
LABEL_36:
    if ( !v29 )
      goto LABEL_37;
LABEL_39:
    v39 = *v20++;
    v84 += v39;
    if ( !v39 )
      goto LABEL_40;
  }
  v76 = v92;
  if ( (v92 & 1) == 0 && v92 )
  {
    _libc_free(*(_QWORD *)v92);
    j_j___libc_free_0(v76);
  }
LABEL_40:
  result = 0;
  if ( v81 )
  {
    result = 1;
    if ( v81 < v79 )
    {
      v77 = *(unsigned int *)(a1 + 16);
      if ( (unsigned int)v77 >= *(_DWORD *)(a1 + 20) )
      {
        sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 16, v23, v24);
        v77 = *(unsigned int *)(a1 + 16);
      }
      v78 = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + 16 * v77);
      *v78 = ((unsigned __int64)(v79 - v81) << 32) | 0xFFFFFFFF;
      v78[1] = (unsigned __int64)"no DWARF register encoding";
      ++*(_DWORD *)(a1 + 16);
      result = 1;
    }
  }
LABEL_42:
  v40 = v91;
  if ( (v91 & 1) == 0 && v91 )
  {
    v90 = result;
    _libc_free(*(_QWORD *)v91);
    j_j___libc_free_0(v40);
    return v90;
  }
  return result;
}
