// Function: sub_2B3D5E0
// Address: 0x2b3d5e0
//
__int64 *__fastcall sub_2B3D5E0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r12
  __int64 v5; // rsi
  __int64 v6; // rax
  char v7; // r10
  __int64 *v8; // rdx
  __int64 *v9; // r9
  __int64 *v10; // r13
  char v11; // r10
  __int64 v12; // rcx
  int v13; // esi
  unsigned int v14; // eax
  __int64 v15; // r8
  __int64 *v16; // r8
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // r11
  __int64 v20; // rsi
  int v21; // r11d
  unsigned int v22; // eax
  __int64 v23; // rdi
  __int64 v24; // rsi
  int v25; // r14d
  unsigned int v26; // eax
  __int64 v27; // rdi
  __int64 v28; // rdi
  int v29; // esi
  int v30; // r14d
  int v31; // r11d
  int v33; // r11d
  int v34; // r14d
  int v35; // r14d
  int v36; // r14d
  int v37; // r14d
  __int64 v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rdi
  int v41; // ecx
  int v42; // r8d
  unsigned int v43; // eax
  __int64 v44; // rsi
  int v45; // ecx
  __int64 v47[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1;
  v5 = a2 - (_QWORD)a1;
  v6 = v5 >> 3;
  if ( v5 >> 5 <= 0 )
  {
LABEL_41:
    switch ( v6 )
    {
      case 2LL:
        v38 = a3 + 80;
        break;
      case 3LL:
        v38 = a3 + 80;
        v47[0] = *v3;
        if ( !sub_2B3D560(a3 + 80, v47) )
          return v3;
        ++v3;
        break;
      case 1LL:
        goto LABEL_48;
      default:
        return (__int64 *)a2;
    }
    v47[0] = *v3;
    if ( !sub_2B3D560(v38, v47) )
      return v3;
    ++v3;
LABEL_48:
    v39 = *v3;
    if ( (*(_BYTE *)(a3 + 88) & 1) != 0 )
    {
      v40 = a3 + 96;
      v41 = 3;
    }
    else
    {
      v45 = *(_DWORD *)(a3 + 104);
      v40 = *(_QWORD *)(a3 + 96);
      if ( !v45 )
        return v3;
      v41 = v45 - 1;
    }
    v42 = 1;
    v43 = v41 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    v44 = *(_QWORD *)(v40 + 72LL * v43);
    if ( v39 != v44 )
    {
      while ( v44 != -4096 )
      {
        v43 = v41 & (v42 + v43);
        v44 = *(_QWORD *)(v40 + 72LL * v43);
        if ( v39 == v44 )
          return (__int64 *)a2;
        ++v42;
      }
      return v3;
    }
    return (__int64 *)a2;
  }
  v7 = *(_BYTE *)(a3 + 88);
  v8 = a1 + 3;
  v9 = a1 + 2;
  v10 = &a1[4 * (v5 >> 5)];
  v11 = v7 & 1;
  while ( 1 )
  {
    v28 = *(v8 - 3);
    if ( v11 )
    {
      v12 = a3 + 96;
      v13 = 3;
    }
    else
    {
      v29 = *(_DWORD *)(a3 + 104);
      v12 = *(_QWORD *)(a3 + 96);
      if ( !v29 )
        return v3;
      v13 = v29 - 1;
    }
    v14 = v13 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v15 = *(_QWORD *)(v12 + 72LL * v14);
    if ( v28 != v15 )
      break;
LABEL_5:
    v16 = v3;
    v17 = *(v8 - 2);
    ++v3;
    if ( v11 )
    {
      v18 = ((unsigned __int8)((unsigned int)v17 >> 9) ^ (unsigned __int8)((unsigned int)v17 >> 4)) & 3;
      v19 = *(_QWORD *)(v12
                      + 72LL
                      * (((unsigned __int8)((unsigned int)v17 >> 9) ^ (unsigned __int8)((unsigned int)v17 >> 4)) & 3));
      if ( v17 != v19 )
        goto LABEL_28;
      v20 = *(v8 - 1);
      v3 = v9;
LABEL_8:
      v21 = 3;
      v22 = ((unsigned __int8)((unsigned int)v20 >> 4) ^ (unsigned __int8)((unsigned int)v20 >> 9)) & 3;
      v23 = *(_QWORD *)(v12
                      + 72LL
                      * (((unsigned __int8)((unsigned int)v20 >> 4) ^ (unsigned __int8)((unsigned int)v20 >> 9)) & 3));
      if ( v23 != v20 )
      {
LABEL_34:
        v36 = 1;
        while ( v23 != -4096 )
        {
          v22 = v21 & (v36 + v22);
          v23 = *(_QWORD *)(v12 + 72LL * v22);
          if ( v23 == v20 )
          {
            v24 = *v8;
            if ( v11 )
            {
              v25 = 3;
            }
            else
            {
              v37 = *(_DWORD *)(a3 + 104);
              if ( !v37 )
                return v8;
              v25 = v37 - 1;
            }
            goto LABEL_10;
          }
          ++v36;
        }
        return v3;
      }
      v24 = *v8;
      v25 = 3;
    }
    else
    {
      v30 = *(_DWORD *)(a3 + 104);
      if ( !v30 )
        return v3;
      v25 = v30 - 1;
      v18 = v25 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v19 = *(_QWORD *)(v12 + 72LL * v18);
      if ( v17 != v19 )
      {
LABEL_28:
        v34 = 1;
        while ( v19 != -4096 )
        {
          v18 = v13 & (v34 + v18);
          v19 = *(_QWORD *)(v12 + 72LL * v18);
          if ( v17 == v19 )
          {
            v20 = *(v8 - 1);
            v3 = v9;
            if ( v11 )
              goto LABEL_8;
            v35 = *(_DWORD *)(a3 + 104);
            if ( v35 )
            {
              v25 = v35 - 1;
              goto LABEL_18;
            }
            return v3;
          }
          ++v34;
        }
        return v3;
      }
      v20 = *(v8 - 1);
      v3 = v9;
LABEL_18:
      v21 = v25;
      v22 = v25 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v23 = *(_QWORD *)(v12 + 72LL * v22);
      if ( v23 != v20 )
        goto LABEL_34;
      v24 = *v8;
    }
LABEL_10:
    v26 = v25 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v27 = *(_QWORD *)(v12 + 72LL * v26);
    if ( v24 != v27 )
    {
      v31 = 1;
      while ( v27 != -4096 )
      {
        v26 = v25 & (v31 + v26);
        v27 = *(_QWORD *)(v12 + 72LL * v26);
        if ( v27 == v24 )
          goto LABEL_11;
        ++v31;
      }
      return v8;
    }
LABEL_11:
    v3 = v16 + 4;
    v8 += 4;
    v9 += 4;
    if ( v10 == v16 + 4 )
    {
      v6 = (a2 - (__int64)v3) >> 3;
      goto LABEL_41;
    }
  }
  v33 = 1;
  while ( v15 != -4096 )
  {
    v14 = v13 & (v33 + v14);
    v15 = *(_QWORD *)(v12 + 72LL * v14);
    if ( v28 == v15 )
      goto LABEL_5;
    ++v33;
  }
  return v3;
}
