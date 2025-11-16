// Function: sub_1E16810
// Address: 0x1e16810
//
__int64 __fastcall sub_1E16810(__int64 a1, int a2, char a3, char a4, __int64 a5)
{
  int v5; // r12d
  char v8; // bl
  unsigned int v9; // edx
  char *v10; // rax
  char v11; // cl
  unsigned __int8 v12; // cl
  int v13; // esi
  __int64 v14; // r11
  __int64 v15; // r10
  __int64 v16; // rdi
  unsigned int v17; // r11d
  unsigned int v18; // r13d
  _WORD *v19; // r14
  _WORD *v20; // r10
  unsigned __int16 v21; // si
  __int16 *v22; // r11
  int v23; // edi
  unsigned __int16 *v24; // r13
  unsigned int v25; // r14d
  unsigned int i; // r10d
  bool v27; // cf
  __int16 *v28; // r14
  __int16 v29; // r11
  int v31; // ecx
  int v32; // r10d
  __int16 *v33; // r13
  __int16 v34; // di
  __int16 *v35; // r13
  unsigned __int16 v36; // r11
  __int16 *v37; // rdi
  __int16 v38; // r10
  __int64 v39; // [rsp+0h] [rbp-40h]
  unsigned int v41; // [rsp+Ch] [rbp-34h]
  __int64 v42; // [rsp+10h] [rbp-30h]

  v5 = *(_DWORD *)(a1 + 40);
  if ( !v5 )
    return 0xFFFFFFFFLL;
  v8 = a4 & (a2 > 0);
  v9 = 0;
  v39 = 24LL * (unsigned int)a2;
  v42 = 4LL * ((unsigned int)a2 >> 5);
  v10 = *(char **)(a1 + 32);
  v41 = a2 & 0x1F;
  while ( 1 )
  {
    v11 = *v10;
    if ( v8 )
    {
      if ( v11 == 12 )
        break;
    }
    if ( v11 )
      goto LABEL_19;
    v12 = v10[3];
    if ( (v12 & 0x10) == 0 )
      goto LABEL_19;
    v13 = *((_DWORD *)v10 + 2);
    if ( a2 == v13 )
      goto LABEL_17;
    if ( a5 )
    {
      if ( a2 <= 0 || v13 <= 0 )
        goto LABEL_19;
      v14 = *(_QWORD *)(a5 + 8);
      v15 = *(_QWORD *)(a5 + 56);
      v16 = v14 + v39;
      if ( a4 )
      {
        v17 = *(_DWORD *)(v14 + 24LL * (unsigned int)v13 + 16);
        v18 = *(_DWORD *)(v16 + 16);
        v23 = a2 * (v18 & 0xF);
        v19 = (_WORD *)(v15 + 2LL * (v17 >> 4));
        v20 = (_WORD *)(v15 + 2LL * (v18 >> 4));
        v21 = *v19 + (v17 & 0xF) * v13;
        v22 = v19 + 1;
        LOWORD(v23) = *v20 + a2 * (v18 & 0xF);
        v24 = v20 + 1;
        v25 = v21;
        for ( i = (unsigned __int16)v23; ; i = (unsigned __int16)v23 )
        {
          v27 = v25 < i;
          if ( v25 == i )
            break;
          while ( v27 )
          {
            v28 = v22 + 1;
            v29 = *v22;
            v21 += v29;
            if ( !v29 )
              goto LABEL_19;
            v22 = v28;
            v25 = v21;
            v27 = v21 < i;
            if ( v21 == i )
              goto LABEL_17;
          }
          v32 = *v24;
          if ( !(_WORD)v32 )
            goto LABEL_19;
          v23 += v32;
          ++v24;
        }
      }
      else
      {
        v33 = (__int16 *)(v15 + 2LL * *(unsigned int *)(v16 + 8));
        v34 = *v33;
        v35 = v33 + 1;
        v36 = v34 + a2;
        if ( !v34 )
          v35 = 0;
LABEL_30:
        v37 = v35;
        if ( !v35 )
          goto LABEL_19;
        while ( v13 != v36 )
        {
          v38 = *v37;
          v35 = 0;
          ++v37;
          if ( !v38 )
            goto LABEL_30;
          v36 += v38;
          if ( !v37 )
            goto LABEL_19;
        }
      }
LABEL_17:
      if ( !a3 )
        return v9;
      goto LABEL_18;
    }
    if ( a2 != v13 )
      goto LABEL_19;
    if ( !a3 )
      return v9;
LABEL_18:
    if ( (((v12 & 0x10) != 0) & (v12 >> 6)) != 0 )
      return v9;
LABEL_19:
    ++v9;
    v10 += 40;
    if ( v5 == v9 )
      return 0xFFFFFFFFLL;
  }
  v31 = *(_DWORD *)(*((_QWORD *)v10 + 3) + v42);
  if ( _bittest(&v31, v41) )
    goto LABEL_19;
  return v9;
}
