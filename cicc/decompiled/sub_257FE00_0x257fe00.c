// Function: sub_257FE00
// Address: 0x257fe00
//
__int64 __fastcall sub_257FE00(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 *v8; // r15
  __int64 v9; // r14
  __int64 *v10; // r13
  __int64 *v11; // r13
  __int64 *v12; // r15
  unsigned int v13; // r8d
  int v15; // eax
  unsigned int v16; // ecx
  __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *i; // rdx
  int v20; // eax
  unsigned int v21; // r8d
  _QWORD *v22; // rax
  _QWORD *j; // rdx
  __int64 v24; // r15
  __int64 v25; // r15
  __int64 v26; // r13
  unsigned __int64 v27; // rdx
  char v28; // al
  __int64 v29; // rsi
  __int64 *v30; // rax
  unsigned int v31; // edx
  unsigned int v32; // eax
  _QWORD *v33; // rdi
  __int64 v34; // r13
  _QWORD *v35; // rax
  __int64 v36; // rsi
  __int64 *v37; // rax
  unsigned int v38; // eax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  _QWORD *k; // rdx
  unsigned int v42; // eax
  int v43; // r14d
  unsigned int v44; // eax
  __int64 v45; // [rsp+0h] [rbp-48h]
  unsigned int v46; // [rsp+14h] [rbp-34h]

  v4 = sub_251BBC0(a2, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), a1, 1, 0, 1);
  v7 = v45;
  if ( v4 )
  {
    v8 = *(__int64 **)(a1 + 240);
    v9 = v4;
    v10 = &v8[2 * *(unsigned int *)(a1 + 256)];
    if ( *(_DWORD *)(a1 + 248) && v8 != v10 )
    {
      while ( 1 )
      {
        if ( *v8 == -4096 )
        {
          if ( v8[1] != -4096 )
            break;
          goto LABEL_56;
        }
        if ( *v8 != -8192 || v8[1] != -8192 )
          break;
LABEL_56:
        v8 += 2;
        if ( v10 == v8 )
          goto LABEL_3;
      }
      if ( v10 != v8 )
      {
        v29 = *v8;
        do
        {
          if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v9 + 168LL))(
                  v9,
                  v29,
                  v8[1],
                  v6,
                  v7) )
            goto LABEL_6;
          v30 = v8 + 2;
          if ( v10 == v8 + 2 )
            break;
          while ( 1 )
          {
            v29 = *v30;
            v8 = v30;
            if ( *v30 != -4096 )
              break;
            if ( v30[1] != -4096 )
              goto LABEL_36;
LABEL_41:
            v30 += 2;
            if ( v10 == v30 )
              goto LABEL_3;
          }
          if ( v29 == -8192 && v30[1] == -8192 )
            goto LABEL_41;
LABEL_36:
          ;
        }
        while ( v10 != v30 );
      }
    }
LABEL_3:
    v11 = *(__int64 **)(a1 + 208);
    v12 = &v11[*(unsigned int *)(a1 + 224)];
    if ( *(_DWORD *)(a1 + 216) && v11 != v12 )
    {
      while ( 1 )
      {
        v36 = *v11;
        if ( *v11 != -8192 && v36 != -4096 )
          break;
        if ( v12 == ++v11 )
          return 1;
      }
      while ( v12 != v11 )
      {
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v9 + 160LL))(
                v9,
                v36,
                v5,
                v6,
                v7) )
          goto LABEL_6;
        v37 = v11 + 1;
        if ( v12 == v11 + 1 )
          break;
        while ( 1 )
        {
          v36 = *v37;
          v11 = v37;
          if ( *v37 != -8192 && v36 != -4096 )
            break;
          if ( v12 == ++v37 )
            return 1;
        }
      }
    }
    return 1;
  }
LABEL_6:
  v15 = *(_DWORD *)(a1 + 248);
  ++*(_QWORD *)(a1 + 232);
  if ( !v15 )
  {
    if ( !*(_DWORD *)(a1 + 252) )
      goto LABEL_13;
    v17 = *(unsigned int *)(a1 + 256);
    if ( (unsigned int)v17 <= 0x40 )
    {
LABEL_10:
      v18 = *(_QWORD **)(a1 + 240);
      for ( i = &v18[2 * v17]; i != v18; *(v18 - 1) = -4096 )
      {
        *v18 = -4096;
        v18 += 2;
      }
      goto LABEL_12;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 240), 16 * v17, 8);
    *(_DWORD *)(a1 + 256) = 0;
    goto LABEL_29;
  }
  v16 = 4 * v15;
  v17 = *(unsigned int *)(a1 + 256);
  if ( (unsigned int)(4 * v15) < 0x40 )
    v16 = 64;
  if ( v16 >= (unsigned int)v17 )
    goto LABEL_10;
  v42 = v15 - 1;
  if ( v42 )
  {
    _BitScanReverse(&v42, v42);
    v43 = 1 << (33 - (v42 ^ 0x1F));
    if ( v43 < 64 )
      v43 = 64;
    if ( v43 == (_DWORD)v17 )
      goto LABEL_89;
  }
  else
  {
    v43 = 64;
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 16 * v17, 8);
  v44 = sub_2544050(v43);
  *(_DWORD *)(a1 + 256) = v44;
  if ( !v44 )
  {
LABEL_29:
    *(_QWORD *)(a1 + 240) = 0;
LABEL_12:
    *(_QWORD *)(a1 + 248) = 0;
    goto LABEL_13;
  }
  *(_QWORD *)(a1 + 240) = sub_C7D670(16LL * v44, 8);
LABEL_89:
  sub_25678A0(a1 + 232);
LABEL_13:
  v20 = *(_DWORD *)(a1 + 216);
  ++*(_QWORD *)(a1 + 200);
  if ( !v20 )
  {
    if ( !*(_DWORD *)(a1 + 220) )
      goto LABEL_19;
    v21 = *(_DWORD *)(a1 + 224);
    if ( v21 <= 0x40 )
      goto LABEL_16;
    sub_C7D6A0(*(_QWORD *)(a1 + 208), 8LL * v21, 8);
    *(_DWORD *)(a1 + 224) = 0;
LABEL_82:
    *(_QWORD *)(a1 + 208) = 0;
LABEL_18:
    *(_QWORD *)(a1 + 216) = 0;
    goto LABEL_19;
  }
  v31 = 4 * v20;
  v21 = *(_DWORD *)(a1 + 224);
  if ( (unsigned int)(4 * v20) < 0x40 )
    v31 = 64;
  if ( v31 >= v21 )
  {
LABEL_16:
    v22 = *(_QWORD **)(a1 + 208);
    for ( j = &v22[v21]; j != v22; ++v22 )
      *v22 = -4096;
    goto LABEL_18;
  }
  v32 = v20 - 1;
  if ( v32 )
  {
    _BitScanReverse(&v32, v32);
    v33 = *(_QWORD **)(a1 + 208);
    v34 = (unsigned int)(1 << (33 - (v32 ^ 0x1F)));
    if ( (int)v34 < 64 )
      v34 = 64;
    if ( (_DWORD)v34 == v21 )
    {
      *(_QWORD *)(a1 + 216) = 0;
      v35 = &v33[v34];
      do
      {
        if ( v33 )
          *v33 = -4096;
        ++v33;
      }
      while ( v35 != v33 );
      goto LABEL_19;
    }
  }
  else
  {
    v33 = *(_QWORD **)(a1 + 208);
    LODWORD(v34) = 64;
  }
  sub_C7D6A0((__int64)v33, 8LL * v21, 8);
  v38 = sub_2544050(v34);
  *(_DWORD *)(a1 + 224) = v38;
  if ( !v38 )
    goto LABEL_82;
  v39 = (_QWORD *)sub_C7D670(8LL * v38, 8);
  v40 = *(unsigned int *)(a1 + 224);
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 208) = v39;
  for ( k = &v39[v40]; k != v39; ++v39 )
  {
    if ( v39 )
      *v39 = -4096;
  }
LABEL_19:
  v24 = *(unsigned int *)(a1 + 112);
  if ( !(_DWORD)v24 )
    return 1;
  v25 = 8 * v24;
  v26 = 0;
  v13 = 1;
  do
  {
    v27 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + v26);
    if ( !*(_DWORD *)(v27 + 24) )
    {
      v46 = v13;
      v28 = sub_257EF80(a1, a2, v27, 0);
      v13 = v46;
      if ( v28 )
        v13 = 0;
    }
    v26 += 8;
  }
  while ( v26 != v25 );
  return v13;
}
