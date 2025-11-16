// Function: sub_1F08650
// Address: 0x1f08650
//
__int64 __fastcall sub_1F08650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // r12
  unsigned __int64 v8; // rcx
  __int64 v9; // rsi
  __int64 result; // rax
  __int64 v11; // rbx
  __int64 v12; // r12
  unsigned __int64 v13; // rax
  unsigned int v14; // esi
  __int64 v15; // r14
  __int64 v16; // r9
  unsigned int v17; // r8d
  __int64 *v18; // rax
  __int64 v19; // rdi
  __int16 v20; // ax
  __int64 v21; // rax
  __int64 v22; // rsi
  char v23; // al
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdx
  unsigned __int16 *v27; // rsi
  unsigned __int16 *i; // rdx
  int v29; // eax
  int v30; // ecx
  __int64 v31; // r10
  unsigned int v32; // edx
  int v33; // edi
  __int64 v34; // r9
  int v35; // r8d
  __int64 *v36; // rsi
  __int64 v37; // r15
  __int64 v38; // r14
  __int64 v39; // rbx
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rcx
  int v43; // r11d
  __int64 *v44; // rcx
  int v45; // ecx
  __int64 v46; // rbx
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // rdi
  int v49; // eax
  int v50; // ecx
  __int64 v51; // r10
  int v52; // r8d
  unsigned int v53; // edx
  __int64 v54; // r9
  __int64 v55; // [rsp+8h] [rbp-58h]
  __int64 v56; // [rsp+8h] [rbp-58h]
  __int64 *v57; // [rsp+10h] [rbp-50h]
  __int64 v58; // [rsp+18h] [rbp-48h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  int v60; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v61[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = *(_QWORD *)(a1 + 48);
  v57 = (__int64 *)(a1 + 48);
  v8 = *(unsigned int *)(a1 + 944);
  v9 = *(_QWORD *)(a1 + 64) - v7;
  if ( v8 <= 0xF0F0F0F0F0F0F0F1LL * (v9 >> 4) )
    goto LABEL_2;
  v37 = *(_QWORD *)(a1 + 56);
  v38 = 272 * v8;
  v56 = v37 - v7;
  if ( *(_DWORD *)(a1 + 944) )
  {
    v59 = sub_22077B0(272 * v8);
    if ( v7 == v37 )
    {
LABEL_64:
      v46 = *(_QWORD *)(a1 + 56);
      v37 = *(_QWORD *)(a1 + 48);
      if ( v46 == v37 )
      {
        v9 = *(_QWORD *)(a1 + 64) - v37;
      }
      else
      {
        do
        {
          v47 = *(_QWORD *)(v37 + 112);
          if ( v47 != v37 + 128 )
            _libc_free(v47);
          v48 = *(_QWORD *)(v37 + 32);
          if ( v48 != v37 + 48 )
            _libc_free(v48);
          v37 += 272;
        }
        while ( v46 != v37 );
        v37 = *(_QWORD *)(a1 + 48);
        v9 = *(_QWORD *)(a1 + 64) - v37;
      }
      goto LABEL_59;
    }
LABEL_41:
    v39 = v59;
    do
    {
      if ( v39 )
      {
        *(_QWORD *)v39 = *(_QWORD *)v7;
        *(_QWORD *)(v39 + 8) = *(_QWORD *)(v7 + 8);
        *(_QWORD *)(v39 + 16) = *(_QWORD *)(v7 + 16);
        v41 = *(_QWORD *)(v7 + 24);
        *(_DWORD *)(v39 + 40) = 0;
        *(_QWORD *)(v39 + 24) = v41;
        *(_QWORD *)(v39 + 32) = v39 + 48;
        *(_DWORD *)(v39 + 44) = 4;
        v42 = *(unsigned int *)(v7 + 40);
        if ( (_DWORD)v42 )
          sub_1F03590(v39 + 32, v7 + 32, a3, v42, a5, a6);
        *(_DWORD *)(v39 + 120) = 0;
        *(_QWORD *)(v39 + 112) = v39 + 128;
        *(_DWORD *)(v39 + 124) = 4;
        v40 = *(unsigned int *)(v7 + 120);
        if ( (_DWORD)v40 )
          sub_1F03590(v39 + 112, v7 + 112, v40, v42, a5, a6);
        *(_DWORD *)(v39 + 192) = *(_DWORD *)(v7 + 192);
        *(_DWORD *)(v39 + 196) = *(_DWORD *)(v7 + 196);
        *(_DWORD *)(v39 + 200) = *(_DWORD *)(v7 + 200);
        *(_DWORD *)(v39 + 204) = *(_DWORD *)(v7 + 204);
        *(_DWORD *)(v39 + 208) = *(_DWORD *)(v7 + 208);
        *(_DWORD *)(v39 + 212) = *(_DWORD *)(v7 + 212);
        *(_DWORD *)(v39 + 216) = *(_DWORD *)(v7 + 216);
        *(_DWORD *)(v39 + 220) = *(_DWORD *)(v7 + 220);
        *(_WORD *)(v39 + 224) = *(_WORD *)(v7 + 224);
        *(_WORD *)(v39 + 226) = *(_WORD *)(v7 + 226);
        *(_WORD *)(v39 + 228) = *(_WORD *)(v7 + 228);
        *(_DWORD *)(v39 + 232) = *(_DWORD *)(v7 + 232);
        a3 = *(_BYTE *)(v7 + 236) & 3;
        *(_BYTE *)(v39 + 236) = a3 | *(_BYTE *)(v39 + 236) & 0xFC;
        *(_DWORD *)(v39 + 240) = *(_DWORD *)(v7 + 240);
        *(_DWORD *)(v39 + 244) = *(_DWORD *)(v7 + 244);
        *(_DWORD *)(v39 + 248) = *(_DWORD *)(v7 + 248);
        *(_DWORD *)(v39 + 252) = *(_DWORD *)(v7 + 252);
        *(_QWORD *)(v39 + 256) = *(_QWORD *)(v7 + 256);
        *(_QWORD *)(v39 + 264) = *(_QWORD *)(v7 + 264);
      }
      v7 += 272;
      v39 += 272;
    }
    while ( v37 != v7 );
    goto LABEL_64;
  }
  v59 = 0;
  if ( v7 != v37 )
    goto LABEL_41;
LABEL_59:
  if ( v37 )
    j_j___libc_free_0(v37, v9);
  *(_QWORD *)(a1 + 48) = v59;
  *(_QWORD *)(a1 + 56) = v59 + v56;
  *(_QWORD *)(a1 + 64) = v59 + v38;
LABEL_2:
  result = *(_QWORD *)(a1 + 936);
  v11 = *(_QWORD *)(a1 + 928);
  v55 = a1 + 952;
  v58 = result;
  if ( result != v11 )
  {
    while ( 1 )
    {
      result = (unsigned int)**(unsigned __int16 **)(v11 + 16) - 12;
      if ( (unsigned __int16)(**(_WORD **)(v11 + 16) - 12) > 1u )
        break;
LABEL_6:
      if ( (*(_BYTE *)v11 & 4) != 0 )
      {
        v11 = *(_QWORD *)(v11 + 8);
        if ( v58 == v11 )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(v11 + 46) & 8) != 0 )
          v11 = *(_QWORD *)(v11 + 8);
        v11 = *(_QWORD *)(v11 + 8);
        if ( v58 == v11 )
          return result;
      }
    }
    v12 = *(_QWORD *)(a1 + 56);
    v61[0] = v11;
    v13 = 0xF0F0F0F0F0F0F0F1LL * ((v12 - *(_QWORD *)(a1 + 48)) >> 4);
    v60 = -252645135 * ((v12 - *(_QWORD *)(a1 + 48)) >> 4);
    if ( v12 == *(_QWORD *)(a1 + 64) )
    {
      sub_1F06C00(v57, (__int64 *)v12, v61, &v60);
      v15 = *(_QWORD *)(a1 + 56);
      v14 = *(_DWORD *)(a1 + 976);
      v12 = v15 - 272;
      if ( v14 )
        goto LABEL_16;
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)(v12 + 8) = v11;
        *(_QWORD *)(v12 + 32) = v12 + 48;
        *(_DWORD *)(v12 + 192) = v13;
        *(_BYTE *)(v12 + 236) &= 0xFCu;
        *(_QWORD *)v12 = 0;
        *(_QWORD *)(v12 + 16) = 0;
        *(_QWORD *)(v12 + 24) = 0;
        *(_QWORD *)(v12 + 40) = 0x400000000LL;
        *(_QWORD *)(v12 + 112) = v12 + 128;
        *(_QWORD *)(v12 + 120) = 0x400000000LL;
        *(_QWORD *)(v12 + 196) = 0;
        *(_QWORD *)(v12 + 204) = 0;
        *(_QWORD *)(v12 + 212) = 0;
        *(_QWORD *)(v12 + 220) = 0;
        *(_WORD *)(v12 + 228) = 0;
        *(_DWORD *)(v12 + 232) = 0;
        *(_QWORD *)(v12 + 240) = 0;
        *(_QWORD *)(v12 + 248) = 0;
        *(_QWORD *)(v12 + 256) = 0;
        *(_QWORD *)(v12 + 264) = 0;
        v12 = *(_QWORD *)(a1 + 56);
      }
      v14 = *(_DWORD *)(a1 + 976);
      v15 = v12 + 272;
      *(_QWORD *)(a1 + 56) = v12 + 272;
      if ( v14 )
      {
LABEL_16:
        v16 = *(_QWORD *)(a1 + 960);
        v17 = (v14 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v18 = (__int64 *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v11 == *v18 )
        {
LABEL_17:
          v18[1] = v12;
          v20 = *(_WORD *)(v11 + 46);
          if ( (v20 & 4) != 0 || (v20 & 8) == 0 )
            v21 = (*(_QWORD *)(*(_QWORD *)(v11 + 16) + 8LL) >> 4) & 1LL;
          else
            LOBYTE(v21) = sub_1E15D00(v11, 0x10u, 1);
          v22 = *(_QWORD *)(v15 - 264);
          v23 = (2 * (v21 & 1)) | *(_BYTE *)(v15 - 44) & 0xFD;
          *(_BYTE *)(v15 - 44) = v23;
          *(_BYTE *)(v15 - 44) = (16 * ((*(_QWORD *)(*(_QWORD *)(v11 + 16) + 8LL) & 0x200000LL) != 0)) | v23 & 0xEF;
          *(_WORD *)(v15 - 46) = sub_1F4BF20(a1 + 632, v22, 1);
          result = sub_1F4B670(a1 + 632);
          if ( (_BYTE)result )
          {
            v24 = *(_QWORD *)(v15 - 248);
            if ( !v24 )
            {
              if ( (unsigned __int8)sub_1F4B670(a1 + 632) )
              {
                v24 = sub_1F4B8B0(a1 + 632, *(_QWORD *)(v15 - 264));
                *(_QWORD *)(v15 - 248) = v24;
              }
              else
              {
                v24 = *(_QWORD *)(v15 - 248);
              }
            }
            v25 = *(unsigned __int16 *)(v24 + 2);
            v26 = *(_QWORD *)(*(_QWORD *)(a1 + 808) + 136LL);
            result = v25 + *(unsigned __int16 *)(v24 + 4);
            v27 = (unsigned __int16 *)(v26 + 4 * result);
            for ( i = (unsigned __int16 *)(v26 + 4 * v25); v27 != i; i += 2 )
            {
              result = *(unsigned int *)(*(_QWORD *)(a1 + 664) + 32LL * *i + 16);
              if ( (_DWORD)result )
              {
                if ( (_DWORD)result == 1 )
                  *(_BYTE *)(v15 - 43) |= 0x40u;
              }
              else
              {
                *(_BYTE *)(v15 - 43) |= 0x80u;
              }
            }
          }
          goto LABEL_6;
        }
        v43 = 1;
        v44 = 0;
        while ( v19 != -8 )
        {
          if ( !v44 && v19 == -16 )
            v44 = v18;
          v17 = (v14 - 1) & (v43 + v17);
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( v11 == *v18 )
            goto LABEL_17;
          ++v43;
        }
        if ( v44 )
          v18 = v44;
        v45 = *(_DWORD *)(a1 + 968);
        ++*(_QWORD *)(a1 + 952);
        v33 = v45 + 1;
        if ( 4 * (v45 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 972) - v33 > v14 >> 3 )
            goto LABEL_55;
          sub_1E51100(v55, v14);
          v49 = *(_DWORD *)(a1 + 976);
          if ( !v49 )
          {
LABEL_91:
            ++*(_DWORD *)(a1 + 968);
            BUG();
          }
          v50 = v49 - 1;
          v51 = *(_QWORD *)(a1 + 960);
          v52 = 1;
          v53 = (v49 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v33 = *(_DWORD *)(a1 + 968) + 1;
          v36 = 0;
          v18 = (__int64 *)(v51 + 16LL * v53);
          v54 = *v18;
          if ( v11 == *v18 )
            goto LABEL_55;
          while ( v54 != -8 )
          {
            if ( !v36 && v54 == -16 )
              v36 = v18;
            v53 = v50 & (v52 + v53);
            v18 = (__int64 *)(v51 + 16LL * v53);
            v54 = *v18;
            if ( v11 == *v18 )
              goto LABEL_55;
            ++v52;
          }
          goto LABEL_36;
        }
LABEL_32:
        sub_1E51100(v55, 2 * v14);
        v29 = *(_DWORD *)(a1 + 976);
        if ( !v29 )
          goto LABEL_91;
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a1 + 960);
        v32 = (v29 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v33 = *(_DWORD *)(a1 + 968) + 1;
        v18 = (__int64 *)(v31 + 16LL * v32);
        v34 = *v18;
        if ( v11 == *v18 )
          goto LABEL_55;
        v35 = 1;
        v36 = 0;
        while ( v34 != -8 )
        {
          if ( v34 == -16 && !v36 )
            v36 = v18;
          v32 = v30 & (v35 + v32);
          v18 = (__int64 *)(v31 + 16LL * v32);
          v34 = *v18;
          if ( v11 == *v18 )
            goto LABEL_55;
          ++v35;
        }
LABEL_36:
        if ( v36 )
          v18 = v36;
LABEL_55:
        *(_DWORD *)(a1 + 968) = v33;
        if ( *v18 != -8 )
          --*(_DWORD *)(a1 + 972);
        *v18 = v11;
        v18[1] = 0;
        goto LABEL_17;
      }
    }
    ++*(_QWORD *)(a1 + 952);
    goto LABEL_32;
  }
  return result;
}
