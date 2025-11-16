// Function: sub_3361C20
// Address: 0x3361c20
//
void __fastcall sub_3361C20(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  int v15; // edi
  unsigned int v16; // eax
  __int64 v17; // r8
  __int64 v18; // rax
  __int32 v19; // r8d
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rcx
  _QWORD *v23; // rsi
  __int32 v24; // eax
  __int64 v25; // rdx
  char v26; // cl
  unsigned int v27; // esi
  __int64 v28; // r9
  int v29; // esi
  unsigned int v30; // edx
  _QWORD *v31; // rax
  __int64 v32; // r10
  _QWORD *v33; // rdi
  int v34; // r11d
  unsigned int v35; // eax
  int v36; // edx
  unsigned int v37; // ecx
  __int64 v38; // rax
  int v39; // r9d
  __int64 v40; // rsi
  int v41; // ecx
  unsigned int v42; // edx
  __int64 v43; // r11
  __int64 v44; // rsi
  int v45; // ecx
  unsigned int v46; // edx
  __int64 v47; // r11
  _QWORD *v48; // rax
  int v49; // r10d
  int v50; // ecx
  int v51; // ecx
  int v52; // r10d
  __int32 v53; // [rsp+8h] [rbp-98h]
  __int32 v54; // [rsp+8h] [rbp-98h]
  __int64 v55; // [rsp+18h] [rbp-88h]
  __int64 v56[4]; // [rsp+20h] [rbp-80h] BYREF
  __m128i v57; // [rsp+40h] [rbp-60h] BYREF
  __int64 v58; // [rsp+50h] [rbp-50h]
  __int64 v59; // [rsp+58h] [rbp-48h]
  __int64 v60; // [rsp+60h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 40);
  v7 = v6 + 16LL * *(unsigned int *)(a2 + 48);
  if ( v6 != v7 )
  {
    while ( (*(_QWORD *)v6 & 6) != 0 )
    {
      v6 += 16;
      if ( v7 == v6 )
        return;
    }
    v12 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_QWORD *)(v12 + 24) )
    {
      if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
      {
        v14 = a3 + 16;
        v15 = 15;
      }
      else
      {
        v13 = *(unsigned int *)(a3 + 24);
        v14 = *(_QWORD *)(a3 + 16);
        if ( !(_DWORD)v13 )
          goto LABEL_32;
        v15 = v13 - 1;
      }
      v16 = v15 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v6 = v14 + 16LL * v16;
      v17 = *(_QWORD *)v6;
      if ( v12 == *(_QWORD *)v6 )
        goto LABEL_8;
      v39 = 1;
      while ( v17 != -4096 )
      {
        v16 = v15 & (v39 + v16);
        v6 = v14 + 16LL * v16;
        v17 = *(_QWORD *)v6;
        if ( v12 == *(_QWORD *)v6 )
          goto LABEL_8;
        ++v39;
      }
      if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
      {
        v38 = 256;
        goto LABEL_33;
      }
      v13 = *(unsigned int *)(a3 + 24);
LABEL_32:
      v38 = 16 * v13;
LABEL_33:
      v6 = v14 + v38;
LABEL_8:
      v18 = *(_QWORD *)(a2 + 120);
      v19 = 0;
      v20 = v18 + 16LL * *(unsigned int *)(a2 + 128);
      if ( v20 != v18 )
      {
        while ( (*(_BYTE *)v18 & 6) != 0 || !*(_DWORD *)(v18 + 8) )
        {
          v18 += 16;
          if ( v20 == v18 )
            goto LABEL_12;
        }
        v19 = *(_DWORD *)(v18 + 8);
      }
      goto LABEL_12;
    }
    v19 = sub_2EC06C0(a1[5], *(_QWORD *)(a2 + 24), byte_3F871B3, 0, a5, a6);
    v26 = *(_BYTE *)(a3 + 8) & 1;
    if ( v26 )
    {
      v28 = a3 + 16;
      v29 = 15;
    }
    else
    {
      v27 = *(_DWORD *)(a3 + 24);
      v28 = *(_QWORD *)(a3 + 16);
      if ( !v27 )
      {
        v35 = *(_DWORD *)(a3 + 8);
        ++*(_QWORD *)a3;
        v33 = 0;
        v36 = (v35 >> 1) + 1;
        goto LABEL_35;
      }
      v29 = v27 - 1;
    }
    v30 = v29 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v31 = (_QWORD *)(v28 + 16LL * v30);
    v32 = *v31;
    if ( a2 == *v31 )
    {
LABEL_12:
      v21 = a1[73];
      v22 = *(_QWORD *)(a1[2] + 8LL);
      memset(v56, 0, 24);
      v55 = 0;
      v23 = sub_2F26260(v21, a4, v56, v22 - 800, v19);
      v24 = *(_DWORD *)(v6 + 8);
      v57.m128i_i64[0] = 0;
      v58 = 0;
      v57.m128i_i32[2] = v24;
      v59 = 0;
      v60 = 0;
      sub_2E8EAD0(v25, (__int64)v23, &v57);
      if ( v56[0] )
        sub_B91220((__int64)v56, v56[0]);
      return;
    }
    v33 = 0;
    v34 = 1;
    while ( v32 != -4096 )
    {
      if ( v32 != -8192 || v33 )
        v31 = v33;
      v30 = v29 & (v34 + v30);
      v32 = *(_QWORD *)(v28 + 16LL * v30);
      if ( a2 == v32 )
        goto LABEL_12;
      ++v34;
      v33 = v31;
      v31 = (_QWORD *)(v28 + 16LL * v30);
    }
    if ( !v33 )
      v33 = v31;
    v35 = *(_DWORD *)(a3 + 8);
    ++*(_QWORD *)a3;
    v36 = (v35 >> 1) + 1;
    if ( v26 )
    {
      v37 = 48;
      v27 = 16;
      goto LABEL_36;
    }
    v27 = *(_DWORD *)(a3 + 24);
LABEL_35:
    v37 = 3 * v27;
LABEL_36:
    if ( 4 * v36 < v37 )
    {
      if ( v27 - *(_DWORD *)(a3 + 12) - v36 > v27 >> 3 )
      {
LABEL_38:
        *(_DWORD *)(a3 + 8) = (2 * (v35 >> 1) + 2) | v35 & 1;
        if ( *v33 != -4096 )
          --*(_DWORD *)(a3 + 12);
        *v33 = a2;
        *((_DWORD *)v33 + 2) = v19;
        goto LABEL_12;
      }
      v54 = v19;
      sub_33617E0(a3, v27);
      v19 = v54;
      if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
      {
        v44 = a3 + 16;
        v45 = 15;
        goto LABEL_51;
      }
      v51 = *(_DWORD *)(a3 + 24);
      v44 = *(_QWORD *)(a3 + 16);
      if ( v51 )
      {
        v45 = v51 - 1;
LABEL_51:
        v46 = v45 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v33 = (_QWORD *)(v44 + 16LL * v46);
        v47 = *v33;
        if ( a2 != *v33 )
        {
          v48 = (_QWORD *)(v44 + 16LL * (v45 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
          v49 = 1;
          v33 = 0;
          while ( v47 != -4096 )
          {
            if ( v47 == -8192 && !v33 )
              v33 = v48;
            v46 = v45 & (v49 + v46);
            v48 = (_QWORD *)(v44 + 16LL * v46);
            v47 = *v48;
            if ( a2 == *v48 )
              goto LABEL_55;
            ++v49;
          }
LABEL_54:
          if ( !v33 )
LABEL_55:
            v33 = v48;
          goto LABEL_48;
        }
        goto LABEL_48;
      }
LABEL_80:
      *(_DWORD *)(a3 + 8) = (2 * (*(_DWORD *)(a3 + 8) >> 1) + 2) | *(_DWORD *)(a3 + 8) & 1;
      BUG();
    }
    v53 = v19;
    sub_33617E0(a3, 2 * v27);
    v19 = v53;
    if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
    {
      v40 = a3 + 16;
      v41 = 15;
    }
    else
    {
      v50 = *(_DWORD *)(a3 + 24);
      v40 = *(_QWORD *)(a3 + 16);
      if ( !v50 )
        goto LABEL_80;
      v41 = v50 - 1;
    }
    v42 = v41 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v33 = (_QWORD *)(v40 + 16LL * v42);
    v43 = *v33;
    if ( a2 != *v33 )
    {
      v48 = (_QWORD *)(v40 + 16LL * (v41 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
      v52 = 1;
      v33 = 0;
      while ( v43 != -4096 )
      {
        if ( !v33 && v43 == -8192 )
          v33 = v48;
        v42 = v41 & (v52 + v42);
        v48 = (_QWORD *)(v40 + 16LL * v42);
        v43 = *v48;
        if ( a2 == *v48 )
          goto LABEL_55;
        ++v52;
      }
      goto LABEL_54;
    }
LABEL_48:
    v35 = *(_DWORD *)(a3 + 8);
    goto LABEL_38;
  }
}
