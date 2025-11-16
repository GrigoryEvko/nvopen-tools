// Function: sub_CB8C00
// Address: 0xcb8c00
//
unsigned __int8 *__fastcall sub_CB8C00(__int64 a1, int a2, int a3)
{
  __int64 v5; // rsi
  unsigned __int8 *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // r14
  int v10; // r9d
  int v11; // r13d
  char *v12; // rdx
  int v13; // r13d
  int v14; // edx
  __int64 v15; // rax
  signed __int64 v16; // rcx
  signed __int64 v17; // rsi
  __int64 v18; // rdx
  int v19; // ecx
  int v20; // esi
  int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // rdx
  signed __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rsi
  signed __int64 v27; // rax
  signed __int64 v28; // rdx
  int v29; // ecx
  __int64 v30; // rdx
  int v31; // r9d
  __int64 v32; // rdx
  int v33; // edx
  unsigned __int8 *v34; // rax
  int v35; // r8d
  unsigned __int8 *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r13
  signed __int64 v39; // rax
  signed __int64 v40; // rsi
  __int64 v41; // rax
  signed __int64 v42; // rcx
  __int64 v43; // rax
  signed __int64 v44; // rdx
  signed __int64 v45; // rsi
  __int64 v46; // rdx
  signed __int64 v47; // rdx
  __int64 v48; // rax
  signed __int64 v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // rax
  signed __int64 v52; // rdx
  signed __int64 v53; // rcx
  signed __int64 v54; // rsi
  __int64 v55; // rcx
  unsigned __int8 *v56; // rcx
  signed __int64 v57; // rdx
  signed __int64 v58; // rcx
  signed __int64 v59; // rsi
  __int64 v60; // rcx
  unsigned __int8 *v61; // rax
  signed __int64 v62; // rsi
  signed __int64 v63; // rsi
  int v64; // r9d
  signed __int64 v65; // rcx
  __int64 v66; // rdx
  signed __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rcx
  int v70; // eax
  unsigned __int8 *v71; // [rsp+0h] [rbp-50h]
  __int64 v72; // [rsp+0h] [rbp-50h]
  __int64 v73; // [rsp+8h] [rbp-48h]
  int v74; // [rsp+10h] [rbp-40h]
  __int64 v75; // [rsp+10h] [rbp-40h]
  __int64 v76; // [rsp+10h] [rbp-40h]
  __int64 v77; // [rsp+10h] [rbp-40h]
  __int64 v78; // [rsp+10h] [rbp-40h]
  int v79; // [rsp+10h] [rbp-40h]
  int v80; // [rsp+10h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 8);
  result = *(unsigned __int8 **)a1;
  v7 = v5 - *(_QWORD *)a1;
  if ( v7 <= 0 )
    goto LABEL_23;
  v8 = *(_QWORD *)(a1 + 40);
  v73 = v8;
  v9 = v8;
  if ( *result == 94 )
  {
    v64 = *(_DWORD *)(a1 + 16);
    *(_QWORD *)a1 = result + 1;
    if ( !v64 )
    {
      v65 = *(_QWORD *)(a1 + 32);
      v66 = v8;
      if ( v8 >= v65 )
      {
        v67 = ((v65 + 1 + ((unsigned __int64)(v65 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v65 + 1) / 2;
        if ( v65 < v67 )
        {
          sub_CB7740(a1, v67);
          v66 = *(_QWORD *)(a1 + 40);
        }
      }
      v68 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = v66 + 1;
      *(_QWORD *)(v68 + 8 * v66) = 402653184;
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 56) + 72LL) |= 1u;
    ++*(_DWORD *)(*(_QWORD *)(a1 + 56) + 76LL);
    v5 = *(_QWORD *)(a1 + 8);
    result = *(unsigned __int8 **)a1;
    v9 = *(_QWORD *)(a1 + 40);
    v7 = v5 - *(_QWORD *)a1;
    if ( v7 <= 0 )
      goto LABEL_42;
  }
  v10 = 1;
  v11 = 0;
  while ( 1 )
  {
    if ( v7 != 1 && (char)*result == a2 && (char)result[1] == a3 )
    {
LABEL_16:
      if ( !v11 )
        goto LABEL_42;
      v14 = *(_DWORD *)(a1 + 16);
      v15 = v9 - 1;
      *(_QWORD *)(a1 + 40) = v9 - 1;
      if ( !v14 )
      {
        v16 = *(_QWORD *)(a1 + 32);
        if ( v15 >= v16 )
        {
          v17 = ((v16 + 1 + ((unsigned __int64)(v16 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v16 + 1) / 2;
          if ( v16 < v17 )
          {
            sub_CB7740(a1, v17);
            v15 = *(_QWORD *)(a1 + 40);
            v9 = v15 + 1;
          }
        }
        v18 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 40) = v9;
        *(_QWORD *)(v18 + 8 * v15) = 0x20000000;
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 56) + 72LL) |= 2u;
      result = *(unsigned __int8 **)(a1 + 56);
      ++*((_DWORD *)result + 20);
      if ( v73 != *(_QWORD *)(a1 + 40) )
        return result;
LABEL_23:
      if ( !*(_DWORD *)(a1 + 16) )
        *(_DWORD *)(a1 + 16) = 14;
      *(_QWORD *)a1 = byte_4F85140;
      *(_QWORD *)(a1 + 8) = byte_4F85140;
      return byte_4F85140;
    }
    v12 = (char *)(result + 1);
    *(_QWORD *)a1 = result + 1;
    v13 = (char)*result;
    if ( v13 != 92 )
      goto LABEL_151;
    v5 -= (__int64)v12;
    if ( v5 <= 0 )
    {
      if ( !*(_DWORD *)(a1 + 16) )
        *(_DWORD *)(a1 + 16) = 5;
      *(_QWORD *)(a1 + 8) = byte_4F85140;
      v12 = (char *)byte_4F85140;
    }
    *(_QWORD *)a1 = v12 + 1;
    v19 = *v12;
    v13 = v19 | 0x100;
    if ( (v19 | 0x100) == 0x128 )
    {
      v51 = ++*(_QWORD *)(*(_QWORD *)(a1 + 56) + 112LL);
      if ( v51 <= 9 )
        *(_QWORD *)(a1 + 8 * v51 + 64) = *(_QWORD *)(a1 + 40);
      if ( !*(_DWORD *)(a1 + 16) )
      {
        v52 = *(_QWORD *)(a1 + 40);
        v53 = *(_QWORD *)(a1 + 32);
        if ( v52 >= v53 )
        {
          v54 = (v53 + 1) / 2 + ((v53 + 1 + ((unsigned __int64)(v53 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
          if ( v53 < v54 )
          {
            v76 = v51;
            sub_CB7740(a1, v54);
            v52 = *(_QWORD *)(a1 + 40);
            v51 = v76;
          }
        }
        v55 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 40) = v52 + 1;
        *(_QWORD *)(v55 + 8 * v52) = v51 | 0x68000000;
      }
      v56 = *(unsigned __int8 **)a1;
      if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) > 0
        && (*(_QWORD *)(a1 + 8) - *(_QWORD *)a1 == 1 || *v56 != 92 || v56[1] != 41) )
      {
        v77 = v51;
        sub_CB8C00(a1, 92, 41);
        v51 = v77;
      }
      if ( v51 <= 9 )
        *(_QWORD *)(a1 + 8 * v51 + 144) = *(_QWORD *)(a1 + 40);
      if ( *(_DWORD *)(a1 + 16) )
      {
        v5 = *(_QWORD *)(a1 + 8);
        v61 = *(unsigned __int8 **)a1;
        if ( v5 - *(_QWORD *)a1 <= 1 )
          goto LABEL_108;
        if ( *v61 != 92 )
        {
LABEL_106:
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 8;
          goto LABEL_108;
        }
      }
      else
      {
        v57 = *(_QWORD *)(a1 + 40);
        v58 = *(_QWORD *)(a1 + 32);
        if ( v57 >= v58 )
        {
          v59 = (v58 + 1) / 2 + ((v58 + 1 + ((unsigned __int64)(v58 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
          if ( v58 < v59 )
          {
            v78 = v51;
            sub_CB7740(a1, v59);
            v57 = *(_QWORD *)(a1 + 40);
            v51 = v78;
          }
        }
        v60 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 40) = v57 + 1;
        *(_QWORD *)(v60 + 8 * v57) = v51 | 0x70000000;
        v5 = *(_QWORD *)(a1 + 8);
        v61 = *(unsigned __int8 **)a1;
        if ( v5 - *(_QWORD *)a1 <= 1 || *v61 != 92 )
          goto LABEL_106;
      }
      if ( v61[1] != 41 )
        goto LABEL_106;
      result = v61 + 2;
      *(_QWORD *)a1 = result;
      goto LABEL_10;
    }
    if ( v13 > 296 )
    {
      if ( v13 == 379 )
      {
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 13;
        goto LABEL_41;
      }
      if ( v13 > 379 )
      {
        if ( v13 == 381 )
          goto LABEL_39;
LABEL_31:
        sub_CB8AB0(a1, (unsigned int)(char)v13);
LABEL_32:
        v5 = *(_QWORD *)(a1 + 8);
        result = *(unsigned __int8 **)a1;
        goto LABEL_10;
      }
      if ( v13 == 297 )
      {
LABEL_39:
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 8;
LABEL_41:
        result = byte_4F85140;
        *(_QWORD *)a1 = byte_4F85140;
        *(_QWORD *)(a1 + 8) = byte_4F85140;
        goto LABEL_42;
      }
      if ( (unsigned int)(v13 - 305) > 8 )
        goto LABEL_31;
      BYTE1(v19) &= ~1u;
      v20 = *(_DWORD *)(a1 + 16);
      v21 = v19 - 48;
      v22 = v21;
      v23 = *(_QWORD *)(a1 + 8LL * v21 + 144);
      if ( v23 )
      {
        if ( !v20 )
        {
          v24 = *(_QWORD *)(a1 + 32);
          v25 = v9;
          if ( v24 <= v9 )
          {
            v63 = (v24 + 1) / 2 + ((v24 + 1 + ((unsigned __int64)(v24 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v24 < v63 )
            {
              v72 = v21;
              v79 = v21;
              sub_CB7740(a1, v63);
              v25 = *(_QWORD *)(a1 + 40);
              v22 = v72;
              v21 = v79;
            }
          }
          v26 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v25 + 1;
          *(_QWORD *)(v26 + 8 * v25) = v21 | 0x38000000;
          v23 = *(_QWORD *)(a1 + 8 * v22 + 144);
        }
        v74 = v21;
        sub_CB77B0((_QWORD *)a1, *(_QWORD *)(a1 + 8 * v22 + 64) + 1LL, v23);
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v27 = *(_QWORD *)(a1 + 40);
          v28 = *(_QWORD *)(a1 + 32);
          v29 = v74;
          if ( v27 >= v28 )
          {
            v62 = (v28 + 1) / 2 + ((v28 + 1 + ((unsigned __int64)(v28 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
            if ( v28 < v62 )
            {
              sub_CB7740(a1, v62);
              v27 = *(_QWORD *)(a1 + 40);
              v29 = v74;
            }
          }
          v30 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v27 + 1;
          *(_QWORD *)(v30 + 8 * v27) = v29 | 0x40000000;
        }
      }
      else
      {
        if ( !v20 )
          *(_DWORD *)(a1 + 16) = 6;
        *(_QWORD *)a1 = byte_4F85140;
        *(_QWORD *)(a1 + 8) = byte_4F85140;
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 56) + 120LL) = 1;
      v5 = *(_QWORD *)(a1 + 8);
      result = *(unsigned __int8 **)a1;
    }
    else
    {
LABEL_151:
      if ( v13 != 46 )
      {
        if ( v13 == 91 )
        {
          sub_CB7E40(a1, v5);
          v5 = *(_QWORD *)(a1 + 8);
          result = *(unsigned __int8 **)a1;
          goto LABEL_10;
        }
        if ( v13 == 42 && !v10 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 13;
          *(_QWORD *)a1 = byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
        }
        goto LABEL_31;
      }
      if ( (*(_BYTE *)(*(_QWORD *)(a1 + 56) + 40LL) & 8) != 0 )
      {
        v5 = *(_QWORD *)(a1 + 8);
        v71 = *(unsigned __int8 **)a1;
        *(_QWORD *)a1 = &unk_3F6AD74;
        *(_QWORD *)(a1 + 8) = &unk_3F6AD77;
        sub_CB7E40(a1, v5);
        result = v71;
        *(_QWORD *)a1 = v71;
        *(_QWORD *)(a1 + 8) = v5;
      }
      else
      {
        if ( *(_DWORD *)(a1 + 16) )
          goto LABEL_32;
        v47 = *(_QWORD *)(a1 + 32);
        v48 = v9;
        if ( v47 <= v9 )
        {
          v49 = (v47 + 1) / 2 + ((v47 + 1 + ((unsigned __int64)(v47 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
          if ( v47 < v49 )
          {
            sub_CB7740(a1, v49);
            v48 = *(_QWORD *)(a1 + 40);
          }
        }
        v50 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 40) = v48 + 1;
        *(_QWORD *)(v50 + 8 * v48) = 671088640;
        v5 = *(_QWORD *)(a1 + 8);
        result = *(unsigned __int8 **)a1;
      }
    }
LABEL_10:
    v7 = v5 - (_QWORD)result;
    if ( v5 - (__int64)result > 0 )
      break;
LABEL_14:
    v9 = *(_QWORD *)(a1 + 40);
    v11 = v13 == 36;
LABEL_15:
    v10 = 0;
    if ( v7 <= 0 )
      goto LABEL_16;
  }
  if ( *result == 42 )
  {
    v31 = *(_DWORD *)(a1 + 16);
    ++result;
    v32 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)a1 = result;
    if ( !v31 )
    {
      sub_CB7820((_QWORD *)a1, 1207959552, v32 - v9 + 1, v9);
      v32 = *(_QWORD *)(a1 + 40);
      v38 = v32 - v9;
      if ( !*(_DWORD *)(a1 + 16) )
      {
        v39 = *(_QWORD *)(a1 + 32);
        if ( v32 >= v39 )
        {
          v40 = (v39 + 1) / 2 + ((v39 + 1 + ((unsigned __int64)(v39 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
          if ( v39 < v40 )
          {
            sub_CB7740(a1, v40);
            v32 = *(_QWORD *)(a1 + 40);
          }
        }
        v41 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 40) = v32 + 1;
        *(_QWORD *)(v41 + 8 * v32) = v38 | 0x50000000;
        if ( !*(_DWORD *)(a1 + 16) )
        {
          sub_CB7820((_QWORD *)a1, 1476395008, *(_QWORD *)(a1 + 40) - v9 + 1, v9);
          v42 = *(_QWORD *)(a1 + 40);
          v11 = *(_DWORD *)(a1 + 16);
          v43 = v42 - v9;
          if ( v11 )
          {
            v5 = *(_QWORD *)(a1 + 8);
            result = *(unsigned __int8 **)a1;
            v9 = *(_QWORD *)(a1 + 40);
            v11 = 0;
            v7 = v5 - *(_QWORD *)a1;
          }
          else
          {
            v44 = *(_QWORD *)(a1 + 32);
            if ( v42 >= v44 )
            {
              v45 = (v44 + 1) / 2 + ((v44 + 1 + ((unsigned __int64)(v44 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
              if ( v44 < v45 )
              {
                v75 = *(_QWORD *)(a1 + 40) - v9;
                sub_CB7740(a1, v45);
                v42 = *(_QWORD *)(a1 + 40);
                v43 = v75;
              }
            }
            v46 = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(a1 + 40) = v42 + 1;
            *(_QWORD *)(v46 + 8 * v42) = v43 | 0x60000000;
            v5 = *(_QWORD *)(a1 + 8);
            result = *(unsigned __int8 **)a1;
            v9 = *(_QWORD *)(a1 + 40);
            v7 = v5 - *(_QWORD *)a1;
          }
          goto LABEL_15;
        }
        v5 = *(_QWORD *)(a1 + 8);
        result = *(unsigned __int8 **)a1;
        v9 = *(_QWORD *)(a1 + 40);
        goto LABEL_58;
      }
      v5 = *(_QWORD *)(a1 + 8);
      result = *(unsigned __int8 **)a1;
    }
    v9 = v32;
LABEL_58:
    v11 = 0;
    v7 = v5 - (_QWORD)result;
    goto LABEL_15;
  }
  if ( v7 == 1 || *result != 92 || result[1] != 123 )
    goto LABEL_14;
  *(_QWORD *)a1 = result + 2;
  v33 = sub_CB7460((unsigned __int8 **)a1);
  v34 = *(unsigned __int8 **)a1;
  v35 = v33;
  if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) > 0 && *v34 == 44 )
  {
    v35 = 256;
    v69 = *(_QWORD *)(a1 + 8) - (_QWORD)(v34 + 1);
    *(_QWORD *)a1 = v34 + 1;
    if ( v69 > 0 && (unsigned int)v34[1] - 48 <= 9 )
    {
      v80 = v33;
      v70 = sub_CB7460((unsigned __int8 **)a1);
      v33 = v80;
      v35 = v70;
      if ( v80 > v70 )
      {
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 10;
        *(_QWORD *)a1 = byte_4F85140;
        *(_QWORD *)(a1 + 8) = byte_4F85140;
      }
    }
  }
  sub_CB7920(a1, v9, v33, v35);
  v5 = *(_QWORD *)(a1 + 8);
  v36 = *(unsigned __int8 **)a1;
  v37 = v5 - *(_QWORD *)a1;
  if ( v37 <= 1 )
  {
    if ( v37 == 1 )
      goto LABEL_130;
LABEL_137:
    if ( !*(_DWORD *)(a1 + 16) )
      *(_DWORD *)(a1 + 16) = 9;
    goto LABEL_108;
  }
  if ( *v36 == 92 && v36[1] == 125 )
  {
    result = v36 + 2;
    v9 = *(_QWORD *)(a1 + 40);
    v11 = 0;
    *(_QWORD *)a1 = result;
    v7 = v5 - (_QWORD)result;
    goto LABEL_15;
  }
LABEL_130:
  while ( v37 == 1 || *v36 != 92 || v36[1] != 125 )
  {
    v37 = v5 - (_QWORD)++v36;
    *(_QWORD *)a1 = v36;
    if ( v5 - (__int64)v36 <= 0 )
      goto LABEL_137;
  }
  if ( !*(_DWORD *)(a1 + 16) )
    *(_DWORD *)(a1 + 16) = 10;
LABEL_108:
  result = byte_4F85140;
  v9 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = byte_4F85140;
  *(_QWORD *)(a1 + 8) = byte_4F85140;
LABEL_42:
  if ( v73 == v9 )
    goto LABEL_23;
  return result;
}
