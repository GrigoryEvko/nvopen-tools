// Function: sub_136BC40
// Address: 0x136bc40
//
__int64 __fastcall sub_136BC40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r10
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // r9d
  unsigned int v9; // edi
  __int64 *v10; // rax
  __int64 v11; // rcx
  int v12; // eax
  __int64 v14; // rax
  int v15; // eax
  int v16; // esi
  __int64 v17; // r9
  unsigned int v18; // ecx
  int v19; // edi
  __int64 v20; // r8
  __int64 v21; // r11
  unsigned int v22; // r15d
  int i; // r14d
  int v24; // r14d
  __int64 *v25; // r11
  __int64 v26; // rcx
  const __m128i *v27; // rsi
  int v28; // ecx
  int v29; // eax
  int v30; // ecx
  __int64 v31; // r8
  __int64 *v32; // r9
  unsigned int v33; // r13d
  int v34; // r10d
  __int64 v35; // rsi
  int v36; // eax
  int v37; // esi
  int v38; // r11d
  __int64 *v39; // r10
  __int64 v40; // [rsp+8h] [rbp-48h]
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+8h] [rbp-48h]
  _DWORD v43[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = a1 + 160;
  v6 = *(_DWORD *)(a1 + 184);
  if ( !v6 )
  {
    v14 = (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3;
    ++*(_QWORD *)(a1 + 160);
    v43[0] = -1431655765 * v14;
    goto LABEL_6;
  }
  v7 = *(_QWORD *)(a1 + 168);
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
  {
LABEL_3:
    v12 = *((_DWORD *)v10 + 2);
    goto LABEL_4;
  }
  v21 = *v10;
  v22 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  for ( i = 1; ; ++i )
  {
    if ( v21 == -8 )
    {
      v24 = 1;
      v25 = 0;
      v43[0] = -1431655765 * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3);
      v26 = *v10;
      if ( *v10 == a2 )
      {
LABEL_14:
        *((_DWORD *)v10 + 2) = v43[0];
        v27 = *(const __m128i **)(a1 + 16);
        if ( v27 == *(const __m128i **)(a1 + 24) )
        {
          v41 = a3;
          sub_1369830((const __m128i **)(a1 + 8), v27);
          a3 = v41;
        }
        else
        {
          if ( v27 )
          {
            v27->m128i_i64[0] = 0;
            v27->m128i_i16[4] = 0;
            v27[1].m128i_i64[0] = 0;
            v27 = *(const __m128i **)(a1 + 16);
          }
          *(_QWORD *)(a1 + 16) = (char *)v27 + 24;
        }
        return sub_1370FC0(a1, v43, a3);
      }
      while ( v26 != -8 )
      {
        if ( !v25 && v26 == -16 )
          v25 = v10;
        v9 = v8 & (v24 + v9);
        v10 = (__int64 *)(v7 + 16LL * v9);
        v26 = *v10;
        if ( *v10 == a2 )
          goto LABEL_14;
        ++v24;
      }
      v28 = *(_DWORD *)(a1 + 176);
      if ( v25 )
        v10 = v25;
      ++*(_QWORD *)(a1 + 160);
      v19 = v28 + 1;
      if ( 4 * (v28 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(a1 + 180) - v19 > v6 >> 3 )
        {
LABEL_8:
          *(_DWORD *)(a1 + 176) = v19;
          if ( *v10 != -8 )
            --*(_DWORD *)(a1 + 180);
          *v10 = a2;
          *((_DWORD *)v10 + 2) = -1;
          goto LABEL_14;
        }
        v42 = a3;
        sub_136BA80(v3, v6);
        v29 = *(_DWORD *)(a1 + 184);
        if ( v29 )
        {
          v30 = v29 - 1;
          v31 = *(_QWORD *)(a1 + 168);
          v32 = 0;
          v33 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v34 = 1;
          v19 = *(_DWORD *)(a1 + 176) + 1;
          a3 = v42;
          v10 = (__int64 *)(v31 + 16LL * v33);
          v35 = *v10;
          if ( *v10 != a2 )
          {
            while ( v35 != -8 )
            {
              if ( v35 == -16 && !v32 )
                v32 = v10;
              v33 = v30 & (v34 + v33);
              v10 = (__int64 *)(v31 + 16LL * v33);
              v35 = *v10;
              if ( *v10 == a2 )
                goto LABEL_8;
              ++v34;
            }
            if ( v32 )
              v10 = v32;
          }
          goto LABEL_8;
        }
LABEL_58:
        ++*(_DWORD *)(a1 + 176);
        BUG();
      }
LABEL_6:
      v40 = a3;
      sub_136BA80(v3, 2 * v6);
      v15 = *(_DWORD *)(a1 + 184);
      if ( v15 )
      {
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 168);
        v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v19 = *(_DWORD *)(a1 + 176) + 1;
        a3 = v40;
        v10 = (__int64 *)(v17 + 16LL * v18);
        v20 = *v10;
        if ( *v10 != a2 )
        {
          v38 = 1;
          v39 = 0;
          while ( v20 != -8 )
          {
            if ( !v39 && v20 == -16 )
              v39 = v10;
            v18 = v16 & (v38 + v18);
            v10 = (__int64 *)(v17 + 16LL * v18);
            v20 = *v10;
            if ( *v10 == a2 )
              goto LABEL_8;
            ++v38;
          }
          if ( v39 )
            v10 = v39;
        }
        goto LABEL_8;
      }
      goto LABEL_58;
    }
    v22 = v8 & (i + v22);
    v21 = *(_QWORD *)(v7 + 16LL * v22);
    if ( v21 == a2 )
      break;
  }
  v36 = 1;
  while ( v11 != -8 )
  {
    v37 = v36 + 1;
    v9 = v8 & (v36 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_3;
    v36 = v37;
  }
  v12 = -1;
LABEL_4:
  v43[0] = v12;
  return sub_1370FC0(a1, v43, a3);
}
