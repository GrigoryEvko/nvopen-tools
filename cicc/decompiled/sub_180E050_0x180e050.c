// Function: sub_180E050
// Address: 0x180e050
//
__int64 __fastcall sub_180E050(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  bool v4; // zf
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // r13
  int v8; // edi
  __int64 v9; // r9
  __int64 v10; // rdx
  unsigned int v11; // r10d
  __int64 *v12; // rax
  __int64 v13; // r14
  __int64 *v14; // r8
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // rcx
  __int64 v20; // r8
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rdi
  int v24; // ecx
  __int64 v25; // rax
  _QWORD *v26; // rdx
  _QWORD *v27; // r15
  __int64 v28; // rax
  unsigned int v29; // r11d
  __int64 **v30; // r9
  int v31; // r11d
  __int64 *v32; // r10
  int v33; // edi
  int v34; // edi
  _QWORD *v35; // [rsp+0h] [rbp-50h]
  __int64 v36[2]; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v37[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a2 + 16) == 53;
  v36[0] = a2;
  if ( v4 )
  {
    if ( (unsigned __int8)sub_180D640(*(_QWORD *)(a1 + 8), a2) )
      return v2;
    return 0;
  }
  v5 = *(unsigned int *)(a1 + 3792);
  v6 = *(_QWORD *)(a1 + 3776);
  v7 = a1 + 3768;
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(a1 + 3768);
    goto LABEL_8;
  }
  v8 = v5 - 1;
  v9 = 1;
  LODWORD(v10) = (v5 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v11 = v10;
  v12 = (__int64 *)(v6 + 16LL * (unsigned int)v10);
  v13 = *v12;
  v14 = (__int64 *)*v12;
  if ( v2 != *v12 )
  {
    while ( v14 != (__int64 *)-8LL )
    {
      v29 = v9 + 1;
      v11 = v8 & (v11 + v9);
      v30 = (__int64 **)(v6 + 16LL * v11);
      v14 = *v30;
      if ( (__int64 *)v2 == *v30 )
      {
        if ( v30 != (__int64 **)(v6 + 16LL * (unsigned int)v5) )
        {
          v12 = (__int64 *)(v6 + 16LL * v11);
          return v12[1];
        }
        v14 = (__int64 *)(v6 + 16LL * (unsigned int)v10);
        goto LABEL_23;
      }
      v9 = v29;
    }
    v10 = v8 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v14 = (__int64 *)(v6 + 16 * v10);
    v13 = *v14;
    if ( v2 == *v14 )
    {
LABEL_66:
      v12 = v14;
      goto LABEL_13;
    }
LABEL_23:
    v9 = 1;
    v12 = 0;
    while ( v13 != -8 )
    {
      if ( !v12 && v13 == -16 )
        v12 = v14;
      LODWORD(v10) = v8 & (v9 + v10);
      v14 = (__int64 *)(v6 + 16LL * (unsigned int)v10);
      v13 = *v14;
      if ( v2 == *v14 )
        goto LABEL_66;
      v9 = (unsigned int)(v9 + 1);
    }
    v24 = *(_DWORD *)(a1 + 3784);
    if ( !v12 )
      v12 = v14;
    ++*(_QWORD *)(a1 + 3768);
    v6 = (unsigned int)(v24 + 1);
    if ( 4 * (int)v6 < (unsigned int)(3 * v5) )
    {
      if ( (int)v5 - *(_DWORD *)(a1 + 3788) - (int)v6 > (unsigned int)v5 >> 3 )
        goto LABEL_10;
LABEL_9:
      sub_180DE90(v7, v5);
      sub_180D3D0(v7, v36, v37);
      v12 = (__int64 *)v37[0];
      v2 = v36[0];
      v6 = (unsigned int)(*(_DWORD *)(a1 + 3784) + 1);
LABEL_10:
      *(_DWORD *)(a1 + 3784) = v6;
      if ( *v12 != -8 )
        --*(_DWORD *)(a1 + 3788);
      v12[1] = 0;
      *v12 = v2;
      v13 = v36[0];
LABEL_13:
      v12[1] = 0;
      v16 = *(_BYTE *)(v13 + 16);
      if ( v16 <= 0x17u )
        return 0;
      v17 = (unsigned int)v16 - 60;
      if ( (unsigned int)v17 > 0xC )
      {
        if ( v16 == 77 )
        {
          v25 = 3LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
          if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
          {
            v26 = *(_QWORD **)(v13 - 8);
            v35 = &v26[v25];
          }
          else
          {
            v35 = (_QWORD *)v13;
            v26 = (_QWORD *)(v13 - v25 * 8);
          }
          if ( v26 != v35 )
          {
            v27 = v26;
            v2 = 0;
            while ( 1 )
            {
              if ( *v27 != v13 )
              {
                v28 = sub_180E050(a1, *v27, v26, v6, v14, v9);
                if ( !v28 || v2 && v28 != v2 )
                  return 0;
                v2 = v28;
              }
              v27 += 3;
              if ( v35 == v27 )
                goto LABEL_16;
            }
          }
          return 0;
        }
        if ( v16 != 56 )
          return 0;
        v2 = sub_180E050(
               a1,
               *(_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)),
               4LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF),
               v6,
               v14,
               v9);
      }
      else
      {
        v2 = sub_180E050(a1, *(_QWORD *)(v13 - 24), v17, v6, v14, v9);
      }
LABEL_16:
      if ( v2 )
      {
        v18 = *(_DWORD *)(a1 + 3792);
        if ( v18 )
        {
          v19 = v36[0];
          v20 = *(_QWORD *)(a1 + 3776);
          v21 = (v18 - 1) & ((LODWORD(v36[0]) >> 9) ^ (LODWORD(v36[0]) >> 4));
          v22 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( *v22 == v36[0] )
          {
LABEL_19:
            v22[1] = v2;
            return v2;
          }
          v31 = 1;
          v32 = 0;
          while ( v23 != -8 )
          {
            if ( v23 == -16 && !v32 )
              v32 = v22;
            v21 = (v18 - 1) & (v31 + v21);
            v22 = (__int64 *)(v20 + 16LL * v21);
            v23 = *v22;
            if ( v36[0] == *v22 )
              goto LABEL_19;
            ++v31;
          }
          v33 = *(_DWORD *)(a1 + 3784);
          if ( v32 )
            v22 = v32;
          ++*(_QWORD *)(a1 + 3768);
          v34 = v33 + 1;
          if ( 4 * v34 < 3 * v18 )
          {
            if ( v18 - *(_DWORD *)(a1 + 3788) - v34 > v18 >> 3 )
            {
LABEL_56:
              *(_DWORD *)(a1 + 3784) = v34;
              if ( *v22 != -8 )
                --*(_DWORD *)(a1 + 3788);
              *v22 = v19;
              v22[1] = 0;
              goto LABEL_19;
            }
LABEL_61:
            sub_180DE90(v7, v18);
            sub_180D3D0(v7, v36, v37);
            v22 = (__int64 *)v37[0];
            v19 = v36[0];
            v34 = *(_DWORD *)(a1 + 3784) + 1;
            goto LABEL_56;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 3768);
        }
        v18 *= 2;
        goto LABEL_61;
      }
      return 0;
    }
LABEL_8:
    LODWORD(v5) = 2 * v5;
    goto LABEL_9;
  }
  if ( v12 == (__int64 *)(v6 + 16 * v5) )
    goto LABEL_13;
  return v12[1];
}
