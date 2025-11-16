// Function: sub_2F96280
// Address: 0x2f96280
//
__int64 __fastcall sub_2F96280(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  unsigned __int64 v8; // rax
  __int64 result; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // r14
  unsigned int v14; // esi
  __int64 v15; // r8
  int v16; // r11d
  __int64 *v17; // rdx
  unsigned int v18; // edi
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 *v21; // rax
  int v22; // eax
  char v23; // al
  __int64 v24; // rsi
  char v25; // al
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rdx
  unsigned __int16 *v29; // rcx
  unsigned __int16 *i; // rdx
  unsigned __int64 v31; // r14
  __int64 v32; // r12
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rcx
  int v36; // eax
  int v37; // eax
  __int64 v38; // rbx
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  int v41; // esi
  int v42; // esi
  __int64 v43; // r10
  __int64 v44; // rcx
  __int64 v45; // r9
  int v46; // r8d
  __int64 *v47; // rdi
  int v48; // ecx
  int v49; // ecx
  __int64 v50; // r9
  __int64 *v51; // rsi
  __int64 v52; // r15
  int v53; // edi
  __int64 v54; // r8
  unsigned __int64 v55; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v56; // [rsp+10h] [rbp-50h]
  __int64 v57; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+18h] [rbp-48h]
  int v59; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v60[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = *(_QWORD *)(a1 + 48);
  v56 = (unsigned __int64 *)(a1 + 48);
  v8 = *(unsigned int *)(a1 + 928);
  if ( v8 <= (*(_QWORD *)(a1 + 64) - v7) >> 8 )
    goto LABEL_2;
  v31 = *(_QWORD *)(a1 + 56);
  v55 = v8 << 8;
  v58 = v31 - v7;
  if ( *(_DWORD *)(a1 + 928) )
  {
    v32 = sub_22077B0(v8 << 8);
    if ( v7 == v31 )
    {
LABEL_61:
      v38 = *(_QWORD *)(a1 + 56);
      v31 = *(_QWORD *)(a1 + 48);
      if ( v38 != v31 )
      {
        do
        {
          v39 = *(_QWORD *)(v31 + 120);
          if ( v39 != v31 + 136 )
            _libc_free(v39);
          v40 = *(_QWORD *)(v31 + 40);
          if ( v40 != v31 + 56 )
            _libc_free(v40);
          v31 += 256LL;
        }
        while ( v38 != v31 );
        v31 = *(_QWORD *)(a1 + 48);
      }
      goto LABEL_58;
    }
LABEL_36:
    v33 = v32;
    do
    {
      if ( v33 )
      {
        *(_QWORD *)v33 = *(_QWORD *)v7;
        *(_QWORD *)(v33 + 8) = *(_QWORD *)(v7 + 8);
        *(_QWORD *)(v33 + 16) = *(_QWORD *)(v7 + 16);
        *(_QWORD *)(v33 + 24) = *(_QWORD *)(v7 + 24);
        v34 = *(_QWORD *)(v7 + 32);
        *(_DWORD *)(v33 + 48) = 0;
        *(_QWORD *)(v33 + 32) = v34;
        *(_QWORD *)(v33 + 40) = v33 + 56;
        *(_DWORD *)(v33 + 52) = 4;
        v35 = *(unsigned int *)(v7 + 48);
        if ( (_DWORD)v35 )
          sub_2F90E90(v33 + 40, v7 + 40, a3, v35, a5, a6);
        *(_DWORD *)(v33 + 128) = 0;
        *(_QWORD *)(v33 + 120) = v33 + 136;
        *(_DWORD *)(v33 + 132) = 4;
        a3 = *(unsigned int *)(v7 + 128);
        if ( (_DWORD)a3 )
          sub_2F90E90(v33 + 120, v7 + 120, a3, v35, a5, a6);
        *(_DWORD *)(v33 + 200) = *(_DWORD *)(v7 + 200);
        *(_DWORD *)(v33 + 204) = *(_DWORD *)(v7 + 204);
        *(_DWORD *)(v33 + 208) = *(_DWORD *)(v7 + 208);
        *(_DWORD *)(v33 + 212) = *(_DWORD *)(v7 + 212);
        *(_DWORD *)(v33 + 216) = *(_DWORD *)(v7 + 216);
        *(_DWORD *)(v33 + 220) = *(_DWORD *)(v7 + 220);
        *(_DWORD *)(v33 + 224) = *(_DWORD *)(v7 + 224);
        *(_DWORD *)(v33 + 228) = *(_DWORD *)(v7 + 228);
        *(_DWORD *)(v33 + 232) = *(_DWORD *)(v7 + 232);
        *(_DWORD *)(v33 + 236) = *(_DWORD *)(v7 + 236);
        *(_DWORD *)(v33 + 240) = *(_DWORD *)(v7 + 240);
        *(_DWORD *)(v33 + 244) = *(_DWORD *)(v7 + 244);
        *(_WORD *)(v33 + 248) = *(_WORD *)(v7 + 248);
        *(_WORD *)(v33 + 250) = *(_WORD *)(v7 + 250);
        *(_WORD *)(v33 + 252) = *(_WORD *)(v7 + 252);
        *(_BYTE *)(v33 + 254) = *(_BYTE *)(v7 + 254);
      }
      v7 += 256;
      v33 += 256;
    }
    while ( v31 != v7 );
    goto LABEL_61;
  }
  v32 = 0;
  if ( v7 != v31 )
    goto LABEL_36;
LABEL_58:
  if ( v31 )
    j_j___libc_free_0(v31);
  *(_QWORD *)(a1 + 48) = v32;
  *(_QWORD *)(a1 + 56) = v32 + v58;
  *(_QWORD *)(a1 + 64) = v55 + v32;
LABEL_2:
  result = *(_QWORD *)(a1 + 920);
  v10 = *(_QWORD *)(a1 + 912);
  v57 = result;
  if ( result != v10 )
  {
    while ( 1 )
    {
      result = *(unsigned __int16 *)(v10 + 68);
      if ( (unsigned __int16)(result - 14) > 4u && (_WORD)result != 24 )
        break;
LABEL_7:
      if ( (*(_BYTE *)v10 & 4) != 0 )
      {
        v10 = *(_QWORD *)(v10 + 8);
        if ( v57 == v10 )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(v10 + 44) & 8) != 0 )
          v10 = *(_QWORD *)(v10 + 8);
        v10 = *(_QWORD *)(v10 + 8);
        if ( v57 == v10 )
          return result;
      }
    }
    v11 = *(_QWORD *)(a1 + 56);
    v60[0] = v10;
    v12 = (v11 - *(_QWORD *)(a1 + 48)) >> 8;
    v59 = v12;
    if ( v11 == *(_QWORD *)(a1 + 64) )
    {
      sub_2F94940(v56, v11, v60, &v59);
      v13 = *(_QWORD *)(a1 + 56);
      v11 = v13 - 256;
    }
    else
    {
      if ( v11 )
      {
        *(_DWORD *)(v11 + 200) = v12;
        *(_QWORD *)(v11 + 40) = v11 + 56;
        *(_QWORD *)v11 = v10;
        *(_QWORD *)(v11 + 8) = 0;
        *(_QWORD *)(v11 + 16) = 0;
        *(_QWORD *)(v11 + 24) = 0;
        *(_QWORD *)(v11 + 32) = 0;
        *(_QWORD *)(v11 + 48) = 0x400000000LL;
        *(_QWORD *)(v11 + 120) = v11 + 136;
        *(_QWORD *)(v11 + 128) = 0x400000000LL;
        *(_QWORD *)(v11 + 204) = 0;
        *(_QWORD *)(v11 + 212) = 0;
        *(_QWORD *)(v11 + 220) = 0;
        *(_QWORD *)(v11 + 228) = 0;
        *(_QWORD *)(v11 + 236) = 0;
        *(_QWORD *)(v11 + 244) = 0;
        *(_WORD *)(v11 + 252) = 0;
        *(_BYTE *)(v11 + 254) = 8;
        v11 = *(_QWORD *)(a1 + 56);
      }
      v13 = v11 + 256;
      *(_QWORD *)(a1 + 56) = v11 + 256;
    }
    v14 = *(_DWORD *)(a1 + 960);
    if ( v14 )
    {
      v15 = *(_QWORD *)(a1 + 944);
      v16 = 1;
      v17 = 0;
      v18 = (v14 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v19 = (__int64 *)(v15 + 16LL * v18);
      v20 = *v19;
      if ( v10 == *v19 )
      {
LABEL_19:
        v21 = v19 + 1;
LABEL_20:
        *v21 = v11;
        v22 = *(_DWORD *)(v10 + 44);
        if ( (v22 & 4) != 0 || (v22 & 8) == 0 )
          v23 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v10 + 16) + 24LL) >> 7;
        else
          v23 = sub_2E88A90(v10, 128, 1);
        v24 = *(_QWORD *)(v13 - 256);
        v25 = (2 * (v23 & 1)) | *(_BYTE *)(v13 - 8) & 0xFD;
        *(_BYTE *)(v13 - 8) = v25;
        *(_BYTE *)(v13 - 8) = (16 * ((*(_QWORD *)(*(_QWORD *)(v10 + 16) + 24LL) & 0x2000000LL) != 0)) | v25 & 0xEF;
        *(_WORD *)(v13 - 4) = sub_2FF8080(a1 + 600, v24, 1);
        result = sub_2FF7B70(a1 + 600);
        if ( (_BYTE)result )
        {
          v26 = *(_QWORD *)(v13 - 240);
          if ( !v26 )
          {
            if ( (unsigned __int8)sub_2FF7B70(a1 + 600) )
            {
              v26 = sub_2FF7DB0(a1 + 600, *(_QWORD *)(v13 - 256));
              *(_QWORD *)(v13 - 240) = v26;
            }
            else
            {
              v26 = *(_QWORD *)(v13 - 240);
            }
          }
          v27 = *(_QWORD *)(*(_QWORD *)(a1 + 792) + 176LL);
          v28 = *(unsigned __int16 *)(v26 + 2);
          v29 = (unsigned __int16 *)(v27 + 6 * (v28 + *(unsigned __int16 *)(v26 + 4)));
          result = 3 * v28;
          for ( i = (unsigned __int16 *)(v27 + 6 * v28); v29 != i; i += 3 )
          {
            result = *(unsigned int *)(*(_QWORD *)(a1 + 632) + 32LL * *i + 16);
            if ( (_DWORD)result )
            {
              if ( (_DWORD)result == 1 )
                *(_BYTE *)(v13 - 7) |= 0x40u;
            }
            else
            {
              *(_BYTE *)(v13 - 7) |= 0x80u;
            }
          }
        }
        goto LABEL_7;
      }
      while ( v20 != -4096 )
      {
        if ( !v17 && v20 == -8192 )
          v17 = v19;
        v18 = (v14 - 1) & (v16 + v18);
        v19 = (__int64 *)(v15 + 16LL * v18);
        v20 = *v19;
        if ( v10 == *v19 )
          goto LABEL_19;
        ++v16;
      }
      if ( !v17 )
        v17 = v19;
      v36 = *(_DWORD *)(a1 + 952);
      ++*(_QWORD *)(a1 + 936);
      v37 = v36 + 1;
      if ( 4 * v37 < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(a1 + 956) - v37 <= v14 >> 3 )
        {
          sub_2F960A0(a1 + 936, v14);
          v48 = *(_DWORD *)(a1 + 960);
          if ( !v48 )
          {
LABEL_96:
            ++*(_DWORD *)(a1 + 952);
            BUG();
          }
          v49 = v48 - 1;
          v50 = *(_QWORD *)(a1 + 944);
          v51 = 0;
          LODWORD(v52) = v49 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v53 = 1;
          v37 = *(_DWORD *)(a1 + 952) + 1;
          v17 = (__int64 *)(v50 + 16LL * (unsigned int)v52);
          v54 = *v17;
          if ( *v17 != v10 )
          {
            while ( v54 != -4096 )
            {
              if ( !v51 && v54 == -8192 )
                v51 = v17;
              v52 = v49 & (unsigned int)(v52 + v53);
              v17 = (__int64 *)(v50 + 16 * v52);
              v54 = *v17;
              if ( v10 == *v17 )
                goto LABEL_54;
              ++v53;
            }
            if ( v51 )
              v17 = v51;
          }
        }
        goto LABEL_54;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 936);
    }
    sub_2F960A0(a1 + 936, 2 * v14);
    v41 = *(_DWORD *)(a1 + 960);
    if ( !v41 )
      goto LABEL_96;
    v42 = v41 - 1;
    v43 = *(_QWORD *)(a1 + 944);
    LODWORD(v44) = v42 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v37 = *(_DWORD *)(a1 + 952) + 1;
    v17 = (__int64 *)(v43 + 16LL * (unsigned int)v44);
    v45 = *v17;
    if ( v10 != *v17 )
    {
      v46 = 1;
      v47 = 0;
      while ( v45 != -4096 )
      {
        if ( v45 == -8192 && !v47 )
          v47 = v17;
        v44 = v42 & (unsigned int)(v44 + v46);
        v17 = (__int64 *)(v43 + 16 * v44);
        v45 = *v17;
        if ( v10 == *v17 )
          goto LABEL_54;
        ++v46;
      }
      if ( v47 )
        v17 = v47;
    }
LABEL_54:
    *(_DWORD *)(a1 + 952) = v37;
    if ( *v17 != -4096 )
      --*(_DWORD *)(a1 + 956);
    *v17 = v10;
    v21 = v17 + 1;
    v17[1] = 0;
    goto LABEL_20;
  }
  return result;
}
