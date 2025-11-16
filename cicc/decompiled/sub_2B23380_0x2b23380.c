// Function: sub_2B23380
// Address: 0x2b23380
//
__int64 *__fastcall sub_2B23380(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r12
  __int64 v5; // r13
  __int64 v7; // rax
  __int64 *v8; // r13
  __int64 v9; // r9
  int v10; // edi
  unsigned int v11; // r10d
  __int64 v12; // r8
  int v13; // edi
  int v14; // edi
  __int64 v15; // r9
  int v16; // r10d
  unsigned int v17; // edx
  __int64 v18; // r8
  __int64 v19; // rsi
  __int64 *v20; // r14
  __int64 v21; // r9
  int v22; // r8d
  unsigned int v23; // edi
  __int64 v24; // rcx
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // r8
  unsigned int v28; // edx
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // r9
  __int64 v32; // rcx
  unsigned int v33; // r8d
  __int64 v34; // rdi
  int v35; // ecx
  int v36; // ecx
  __int64 v37; // r8
  unsigned int v38; // edx
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // r8
  __int64 v42; // rcx
  unsigned int v43; // edx
  unsigned int v44; // r9d
  __int64 v45; // rdi
  int v46; // eax
  int v47; // eax
  __int64 v48; // rdi
  unsigned int v49; // edx
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rsi
  int v53; // edi
  __int64 v55; // rdx
  int v56; // r8d
  int v57; // r9d
  int v58; // r10d
  __int64 v59; // rdx
  int v60; // r11d
  int v61; // r10d
  __int64 v62; // rdx
  int v63; // r10d
  __int64 v64; // rdx
  int v65; // r9d
  int v66; // r8d
  __int64 v67; // [rsp+0h] [rbp-30h] BYREF
  __int64 v68; // [rsp+8h] [rbp-28h]

  v4 = a1;
  v5 = (a2 - (__int64)a1) >> 7;
  v7 = (a2 - (__int64)a1) >> 5;
  v67 = a3;
  v68 = a4;
  if ( v5 > 0 )
  {
    v8 = &a1[16 * v5];
    while ( 1 )
    {
      v51 = v67;
      v52 = *v4;
      v24 = *(_BYTE *)(v67 + 88) & 1;
      if ( (*(_BYTE *)(v67 + 88) & 1) != 0 )
      {
        v9 = v67 + 96;
        v10 = 3;
      }
      else
      {
        v53 = *(_DWORD *)(v67 + 104);
        v9 = *(_QWORD *)(v67 + 96);
        if ( !v53 )
        {
          if ( *(_BYTE *)v52 != 90 )
          {
            v19 = v4[4];
            v20 = v4 + 4;
LABEL_30:
            if ( *(_BYTE *)v19 != 90 )
            {
              v30 = v4[8];
              v20 = v4 + 8;
              goto LABEL_32;
            }
LABEL_62:
            v62 = *(_QWORD *)(v19 + 16);
            if ( v62 && !*(_QWORD *)(v62 + 8) )
            {
              if ( (unsigned __int8)sub_B19060(v51 + 768, v19, v62, v24) )
                return v20;
              v51 = v67;
            }
            goto LABEL_12;
          }
          v55 = *(_QWORD *)(v52 + 16);
          if ( !v55 || *(_QWORD *)(v55 + 8) )
          {
            v19 = v4[4];
            v20 = v4 + 4;
            goto LABEL_40;
          }
LABEL_57:
          if ( (unsigned __int8)sub_B19060(v67 + 768, v52, v55, v24) )
            return v4;
          v51 = v67;
          v24 = *(_BYTE *)(v67 + 88) & 1;
          goto LABEL_7;
        }
        v10 = v53 - 1;
      }
      v11 = v10 & (((unsigned int)v52 >> 4) ^ ((unsigned int)v52 >> 9));
      v12 = *(_QWORD *)(v9 + 72LL * v11);
      if ( v52 != v12 )
      {
        v60 = 1;
        while ( v12 != -4096 )
        {
          v11 = v10 & (v60 + v11);
          v12 = *(_QWORD *)(v9 + 72LL * v11);
          if ( v52 == v12 )
            goto LABEL_5;
          ++v60;
        }
        if ( *(_BYTE *)v52 != 90 )
          goto LABEL_7;
        v55 = *(_QWORD *)(v52 + 16);
        if ( !v55 || *(_QWORD *)(v55 + 8) )
          goto LABEL_7;
        goto LABEL_57;
      }
LABEL_5:
      v13 = *(_DWORD *)(v68 + 24);
      if ( !v13 )
        return v4;
      v14 = v13 - 1;
      v15 = *(_QWORD *)(v68 + 8);
      v16 = 1;
      v17 = v14 & (((unsigned int)v52 >> 4) ^ ((unsigned int)v52 >> 9));
      v18 = *(_QWORD *)(v15 + 16LL * v17);
      if ( v52 != v18 )
      {
        while ( v18 != -4096 )
        {
          v17 = v14 & (v16 + v17);
          v18 = *(_QWORD *)(v15 + 16LL * v17);
          if ( v52 == v18 )
            goto LABEL_7;
          ++v16;
        }
        return v4;
      }
LABEL_7:
      v19 = v4[4];
      v20 = v4 + 4;
      if ( (_BYTE)v24 )
      {
        v21 = v51 + 96;
        v22 = 3;
        goto LABEL_9;
      }
LABEL_40:
      v56 = *(_DWORD *)(v51 + 104);
      v21 = *(_QWORD *)(v51 + 96);
      if ( !v56 )
        goto LABEL_30;
      v22 = v56 - 1;
LABEL_9:
      v23 = v22 & (((unsigned int)v19 >> 4) ^ ((unsigned int)v19 >> 9));
      v24 = *(_QWORD *)(v21 + 72LL * v23);
      if ( v19 != v24 )
      {
        v61 = 1;
        while ( v24 != -4096 )
        {
          v23 = v22 & (v61 + v23);
          v24 = *(_QWORD *)(v21 + 72LL * v23);
          if ( v19 == v24 )
            goto LABEL_10;
          ++v61;
        }
        if ( *(_BYTE *)v19 != 90 )
          goto LABEL_12;
        goto LABEL_62;
      }
LABEL_10:
      v25 = *(_DWORD *)(v68 + 24);
      if ( !v25 )
        return v20;
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v68 + 8);
      v28 = v26 & (((unsigned int)v19 >> 4) ^ ((unsigned int)v19 >> 9));
      v29 = *(_QWORD *)(v27 + 16LL * v28);
      if ( v19 != v29 )
      {
        v57 = 1;
        while ( v29 != -4096 )
        {
          v28 = v26 & (v57 + v28);
          v29 = *(_QWORD *)(v27 + 16LL * v28);
          if ( v19 == v29 )
            goto LABEL_12;
          ++v57;
        }
        return v20;
      }
LABEL_12:
      v30 = v4[8];
      v20 = v4 + 8;
      v31 = v51 + 96;
      v32 = 3;
      if ( (*(_BYTE *)(v51 + 88) & 1) != 0 )
        goto LABEL_13;
LABEL_32:
      v32 = *(unsigned int *)(v51 + 104);
      v31 = *(_QWORD *)(v51 + 96);
      if ( (_DWORD)v32 )
      {
        v32 = (unsigned int)(v32 - 1);
LABEL_13:
        v33 = v32 & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
        v34 = *(_QWORD *)(v31 + 72LL * v33);
        if ( v30 == v34 )
        {
LABEL_14:
          v35 = *(_DWORD *)(v68 + 24);
          if ( !v35 )
            return v20;
          v36 = v35 - 1;
          v37 = *(_QWORD *)(v68 + 8);
          v38 = v36 & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
          v39 = *(_QWORD *)(v37 + 16LL * v38);
          if ( v30 != v39 )
          {
            v65 = 1;
            while ( v39 != -4096 )
            {
              v38 = v36 & (v65 + v38);
              v39 = *(_QWORD *)(v37 + 16LL * v38);
              if ( v30 == v39 )
                goto LABEL_16;
              ++v65;
            }
            return v20;
          }
        }
        else
        {
          v63 = 1;
          while ( v34 != -4096 )
          {
            v33 = v32 & (v63 + v33);
            v34 = *(_QWORD *)(v31 + 72LL * v33);
            if ( v34 == v30 )
              goto LABEL_14;
            ++v63;
          }
          if ( *(_BYTE *)v30 == 90 )
          {
LABEL_69:
            v64 = *(_QWORD *)(v30 + 16);
            if ( v64 && !*(_QWORD *)(v64 + 8) )
            {
              if ( (unsigned __int8)sub_B19060(v51 + 768, v30, v64, v32) )
                return v20;
              v51 = v67;
            }
          }
        }
LABEL_16:
        v40 = v4[12];
        v20 = v4 + 12;
        v41 = v51 + 96;
        v42 = 3;
        if ( (*(_BYTE *)(v51 + 88) & 1) != 0 )
          goto LABEL_17;
        goto LABEL_36;
      }
      if ( *(_BYTE *)v30 == 90 )
        goto LABEL_69;
      v40 = v4[12];
      v20 = v4 + 12;
LABEL_36:
      v42 = *(unsigned int *)(v51 + 104);
      v41 = *(_QWORD *)(v51 + 96);
      if ( !(_DWORD)v42 )
        goto LABEL_47;
      v42 = (unsigned int)(v42 - 1);
LABEL_17:
      v43 = ((unsigned int)v40 >> 4) ^ ((unsigned int)v40 >> 9);
      v44 = v43 & v42;
      v45 = *(_QWORD *)(v41 + 72LL * (v43 & (unsigned int)v42));
      if ( v40 == v45 )
      {
LABEL_18:
        v46 = *(_DWORD *)(v68 + 24);
        if ( !v46 )
          return v20;
        v47 = v46 - 1;
        v48 = *(_QWORD *)(v68 + 8);
        v49 = v47 & v43;
        v50 = *(_QWORD *)(v48 + 16LL * v49);
        if ( v50 != v40 )
        {
          v66 = 1;
          while ( v50 != -4096 )
          {
            v49 = v47 & (v66 + v49);
            v50 = *(_QWORD *)(v48 + 16LL * v49);
            if ( v40 == v50 )
              goto LABEL_20;
            ++v66;
          }
          return v20;
        }
        goto LABEL_20;
      }
      v58 = 1;
      while ( v45 != -4096 )
      {
        v44 = v42 & (v58 + v44);
        v45 = *(_QWORD *)(v41 + 72LL * v44);
        if ( v40 == v45 )
          goto LABEL_18;
        ++v58;
      }
LABEL_47:
      if ( *(_BYTE *)v40 == 90 )
      {
        v59 = *(_QWORD *)(v40 + 16);
        if ( v59 )
        {
          if ( !*(_QWORD *)(v59 + 8) && (unsigned __int8)sub_B19060(v51 + 768, v40, v59, v42) )
            return v20;
        }
      }
LABEL_20:
      v4 += 16;
      if ( v4 == v8 )
      {
        v7 = (a2 - (__int64)v4) >> 5;
        break;
      }
    }
  }
  switch ( v7 )
  {
    case 2LL:
LABEL_90:
      if ( !(unsigned __int8)sub_2B22B70(&v67, *v4) )
        return v4;
      v4 += 4;
LABEL_92:
      if ( (unsigned __int8)sub_2B22B70(&v67, *v4) )
        return (__int64 *)a2;
      return v4;
    case 3LL:
      if ( !(unsigned __int8)sub_2B22B70(&v67, *v4) )
        return v4;
      v4 += 4;
      goto LABEL_90;
    case 1LL:
      goto LABEL_92;
  }
  return (__int64 *)a2;
}
