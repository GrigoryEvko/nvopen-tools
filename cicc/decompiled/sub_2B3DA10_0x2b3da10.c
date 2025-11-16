// Function: sub_2B3DA10
// Address: 0x2b3da10
//
__int64 *__fastcall sub_2B3DA10(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r9
  __int64 v4; // rax
  __int64 *v6; // r12
  _QWORD *v8; // rax
  __int64 *v9; // r8
  __int64 *v10; // rsi
  __int64 *v11; // r9
  __int64 v12; // r11
  _BYTE *v13; // rdx
  _BYTE *v14; // rdx
  _BYTE *v15; // rdx
  _BYTE *v16; // rdx
  __int64 v17; // r10
  int v18; // ecx
  unsigned int v19; // edi
  _BYTE *v20; // r14
  __int64 v22; // r10
  int v23; // ecx
  unsigned int v24; // edi
  _BYTE *v25; // r14
  __int64 v26; // rdi
  int v27; // ecx
  __int64 v28; // r10
  int v29; // ecx
  unsigned int v30; // edi
  _BYTE *v31; // r14
  __int64 v32; // r10
  int v33; // ecx
  unsigned int v34; // edi
  _BYTE *v35; // r14
  int v36; // ecx
  int v37; // ecx
  int v38; // r15d
  int v39; // r15d
  int v40; // r15d
  int v41; // r15d
  __int64 v42[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = (a2 - (__int64)a1) >> 5;
  v4 = (a2 - (__int64)a1) >> 3;
  v6 = a1;
  if ( v3 > 0 )
  {
    v8 = a1 + 3;
    v9 = a1 + 2;
    v10 = a1 + 1;
    v11 = &a1[4 * v3];
    v12 = a3 + 96;
    while ( 1 )
    {
      v16 = (_BYTE *)*(v8 - 3);
      if ( *v16 == 61 )
      {
        if ( (*(_BYTE *)(a3 + 88) & 1) != 0 )
        {
          v17 = v12;
          v18 = 3;
        }
        else
        {
          v17 = *(_QWORD *)(a3 + 96);
          v27 = *(_DWORD *)(a3 + 104);
          v26 = v17;
          if ( !v27 )
          {
            if ( *(_BYTE *)*(v8 - 2) == 61 )
            {
LABEL_48:
              if ( *(_BYTE *)*(v8 - 1) == 61 )
                goto LABEL_6;
              goto LABEL_5;
            }
            v14 = (_BYTE *)*(v8 - 1);
            if ( *v14 == 61 )
              goto LABEL_18;
            goto LABEL_5;
          }
          v18 = v27 - 1;
        }
        v19 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v20 = *(_BYTE **)(v17 + 72LL * v19);
        if ( v16 == v20 )
          return v6;
        v38 = 1;
        while ( v20 != (_BYTE *)-4096LL )
        {
          v19 = v18 & (v38 + v19);
          v20 = *(_BYTE **)(v17 + 72LL * v19);
          if ( v16 == v20 )
            return v6;
          ++v38;
        }
      }
      v13 = (_BYTE *)*(v8 - 2);
      if ( *v13 == 61 )
      {
        if ( (*(_BYTE *)(a3 + 88) & 1) != 0 )
        {
          v22 = v12;
          v23 = 3;
        }
        else
        {
          v36 = *(_DWORD *)(a3 + 104);
          v22 = *(_QWORD *)(a3 + 96);
          if ( !v36 )
            goto LABEL_48;
          v23 = v36 - 1;
        }
        v24 = v23 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v25 = *(_BYTE **)(v22 + 72LL * v24);
        if ( v13 == v25 )
          return v10;
        v39 = 1;
        while ( v25 != (_BYTE *)-4096LL )
        {
          v24 = v23 & (v39 + v24);
          v25 = *(_BYTE **)(v22 + 72LL * v24);
          if ( v13 == v25 )
            return v10;
          ++v39;
        }
      }
      v14 = (_BYTE *)*(v8 - 1);
      if ( *v14 == 61 )
      {
        if ( (*(_BYTE *)(a3 + 88) & 1) != 0 )
        {
          v28 = v12;
          v29 = 3;
        }
        else
        {
          v26 = *(_QWORD *)(a3 + 96);
          v27 = *(_DWORD *)(a3 + 104);
LABEL_18:
          v28 = v26;
          if ( !v27 )
            goto LABEL_6;
          v29 = v27 - 1;
        }
        v30 = v29 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v31 = *(_BYTE **)(v28 + 72LL * v30);
        if ( v31 == v14 )
          return v9;
        v40 = 1;
        while ( v31 != (_BYTE *)-4096LL )
        {
          v30 = v29 & (v40 + v30);
          v31 = *(_BYTE **)(v28 + 72LL * v30);
          if ( v31 == v14 )
            return v9;
          ++v40;
        }
      }
LABEL_5:
      v15 = (_BYTE *)*v8;
      if ( *(_BYTE *)*v8 == 61 )
      {
        if ( (*(_BYTE *)(a3 + 88) & 1) != 0 )
        {
          v32 = v12;
          v33 = 3;
LABEL_24:
          v34 = v33 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v35 = *(_BYTE **)(v32 + 72LL * v34);
          if ( v15 == v35 )
            return v8;
          v41 = 1;
          while ( v35 != (_BYTE *)-4096LL )
          {
            v34 = v33 & (v41 + v34);
            v35 = *(_BYTE **)(v32 + 72LL * v34);
            if ( v15 == v35 )
              return v8;
            ++v41;
          }
          goto LABEL_6;
        }
        v37 = *(_DWORD *)(a3 + 104);
        v32 = *(_QWORD *)(a3 + 96);
        if ( v37 )
        {
          v33 = v37 - 1;
          goto LABEL_24;
        }
      }
LABEL_6:
      v6 += 4;
      v8 += 4;
      v9 += 4;
      v10 += 4;
      if ( v11 == v6 )
      {
        v4 = (a2 - (__int64)v6) >> 3;
        break;
      }
    }
  }
  if ( v4 == 2 )
  {
LABEL_40:
    if ( *(_BYTE *)*v6 == 61 )
    {
      v42[0] = *v6;
      if ( sub_2B3D560(a3 + 80, v42) )
        return v6;
    }
    ++v6;
    goto LABEL_42;
  }
  if ( v4 == 3 )
  {
    if ( *(_BYTE *)*v6 == 61 )
    {
      v42[0] = *v6;
      if ( sub_2B3D560(a3 + 80, v42) )
        return v6;
    }
    ++v6;
    goto LABEL_40;
  }
  if ( v4 != 1 )
    return (__int64 *)a2;
LABEL_42:
  if ( *(_BYTE *)*v6 != 61 )
    return (__int64 *)a2;
  v42[0] = *v6;
  if ( !sub_2B3D560(a3 + 80, v42) )
    return (__int64 *)a2;
  return v6;
}
