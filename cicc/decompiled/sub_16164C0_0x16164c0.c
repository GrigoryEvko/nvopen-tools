// Function: sub_16164C0
// Address: 0x16164c0
//
_QWORD *__fastcall sub_16164C0(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rax
  char v5; // di
  __int64 v6; // rcx
  int v7; // edi
  __int64 v8; // r8
  int v9; // r9d
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // rsi
  _QWORD *result; // rax
  __int64 v14; // r9
  __int64 v15; // r13
  __int64 v16; // rdi
  int v17; // esi
  unsigned int v18; // edx
  __int64 v19; // r8
  __int64 v20; // r14
  char v21; // cl
  unsigned int v22; // esi
  unsigned int v23; // edx
  int v24; // edi
  unsigned int v25; // r8d
  unsigned int v26; // esi
  int v27; // r11d
  _QWORD *v28; // r10
  unsigned int v29; // edx
  int v30; // r8d
  unsigned int v31; // r9d
  int v32; // eax
  __int64 v33; // rcx
  int v34; // esi
  unsigned int v35; // edx
  __int64 v36; // rdi
  int v37; // ecx
  __int64 v38; // rsi
  int v39; // ecx
  unsigned int v40; // edx
  __int64 v41; // rdi
  int v42; // r10d
  _QWORD *v43; // r8
  int v44; // r11d
  __int64 *v45; // r10
  int v46; // r10d
  __int64 v47; // [rsp+8h] [rbp-48h]
  __int64 v48; // [rsp+8h] [rbp-48h]
  __int64 v49; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v50; // [rsp+18h] [rbp-38h] BYREF

  (*(void (__fastcall **)(_QWORD *))(*a2 + 152LL))(a2);
  v4 = *(unsigned int *)(a1 + 264);
  if ( (unsigned int)v4 >= *(_DWORD *)(a1 + 268) )
  {
    sub_16CD150(a1 + 256, a1 + 272, 0, 8);
    v4 = *(unsigned int *)(a1 + 264);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 256) + 8 * v4) = a2;
  v5 = *(_BYTE *)(a1 + 408);
  ++*(_DWORD *)(a1 + 264);
  v6 = a2[2];
  v49 = v6;
  v7 = v5 & 1;
  if ( v7 )
  {
    v8 = a1 + 416;
    v9 = 7;
  }
  else
  {
    v26 = *(_DWORD *)(a1 + 424);
    v8 = *(_QWORD *)(a1 + 416);
    if ( !v26 )
    {
      v29 = *(_DWORD *)(a1 + 408);
      ++*(_QWORD *)(a1 + 400);
      v11 = 0;
      v30 = (v29 >> 1) + 1;
LABEL_31:
      v31 = 3 * v26;
      goto LABEL_32;
    }
    v9 = v26 - 1;
  }
  v10 = v9 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v11 = (__int64 *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( v6 == *v11 )
    goto LABEL_6;
  v44 = 1;
  v45 = 0;
  while ( v12 != -4 )
  {
    if ( v12 == -8 && !v45 )
      v45 = v11;
    v10 = v9 & (v44 + v10);
    v11 = (__int64 *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( v6 == *v11 )
      goto LABEL_6;
    ++v44;
  }
  v29 = *(_DWORD *)(a1 + 408);
  v31 = 24;
  v26 = 8;
  if ( v45 )
    v11 = v45;
  ++*(_QWORD *)(a1 + 400);
  v30 = (v29 >> 1) + 1;
  if ( !(_BYTE)v7 )
  {
    v26 = *(_DWORD *)(a1 + 424);
    goto LABEL_31;
  }
LABEL_32:
  if ( 4 * v30 >= v31 )
  {
    v26 *= 2;
    goto LABEL_53;
  }
  if ( v26 - *(_DWORD *)(a1 + 412) - v30 <= v26 >> 3 )
  {
LABEL_53:
    sub_16160D0(a1 + 400, v26);
    sub_1610E60(a1 + 400, &v49, &v50);
    v11 = v50;
    v6 = v49;
    v29 = *(_DWORD *)(a1 + 408);
  }
  *(_DWORD *)(a1 + 408) = (2 * (v29 >> 1) + 2) | v29 & 1;
  if ( *v11 != -4 )
    --*(_DWORD *)(a1 + 412);
  *v11 = v6;
  v12 = v49;
  v11[1] = 0;
LABEL_6:
  v11[1] = (__int64)a2;
  result = (_QWORD *)sub_1614F20(a1, v12);
  v14 = result[7];
  v15 = result[6];
  if ( v15 != v14 )
  {
    while ( 1 )
    {
      v20 = *(_QWORD *)(*(_QWORD *)v15 + 32LL);
      v21 = *(_BYTE *)(a1 + 408) & 1;
      if ( v21 )
      {
        v16 = a1 + 416;
        v17 = 7;
      }
      else
      {
        v22 = *(_DWORD *)(a1 + 424);
        v16 = *(_QWORD *)(a1 + 416);
        if ( !v22 )
        {
          v23 = *(_DWORD *)(a1 + 408);
          ++*(_QWORD *)(a1 + 400);
          result = 0;
          v24 = (v23 >> 1) + 1;
LABEL_15:
          v25 = 3 * v22;
          goto LABEL_16;
        }
        v17 = v22 - 1;
      }
      v18 = v17 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      result = (_QWORD *)(v16 + 16LL * v18);
      v19 = *result;
      if ( v20 == *result )
      {
LABEL_10:
        v15 += 8;
        result[1] = a2;
        if ( v14 == v15 )
          return result;
      }
      else
      {
        v27 = 1;
        v28 = 0;
        while ( v19 != -4 )
        {
          if ( !v28 && v19 == -8 )
            v28 = result;
          v18 = v17 & (v27 + v18);
          result = (_QWORD *)(v16 + 16LL * v18);
          v19 = *result;
          if ( v20 == *result )
            goto LABEL_10;
          ++v27;
        }
        v23 = *(_DWORD *)(a1 + 408);
        v25 = 24;
        v22 = 8;
        if ( v28 )
          result = v28;
        ++*(_QWORD *)(a1 + 400);
        v24 = (v23 >> 1) + 1;
        if ( !v21 )
        {
          v22 = *(_DWORD *)(a1 + 424);
          goto LABEL_15;
        }
LABEL_16:
        if ( 4 * v24 >= v25 )
        {
          v47 = v14;
          sub_16160D0(a1 + 400, 2 * v22);
          v14 = v47;
          if ( (*(_BYTE *)(a1 + 408) & 1) != 0 )
          {
            v33 = a1 + 416;
            v34 = 7;
          }
          else
          {
            v32 = *(_DWORD *)(a1 + 424);
            v33 = *(_QWORD *)(a1 + 416);
            if ( !v32 )
              goto LABEL_82;
            v34 = v32 - 1;
          }
          v35 = v34 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          result = (_QWORD *)(v33 + 16LL * v35);
          v36 = *result;
          if ( v20 != *result )
          {
            v46 = 1;
            v43 = 0;
            while ( v36 != -4 )
            {
              if ( !v43 && v36 == -8 )
                v43 = result;
              v35 = v34 & (v46 + v35);
              result = (_QWORD *)(v33 + 16LL * v35);
              v36 = *result;
              if ( v20 == *result )
                goto LABEL_41;
              ++v46;
            }
LABEL_48:
            if ( v43 )
              result = v43;
          }
LABEL_41:
          v23 = *(_DWORD *)(a1 + 408);
          goto LABEL_18;
        }
        if ( v22 - *(_DWORD *)(a1 + 412) - v24 <= v22 >> 3 )
        {
          v48 = v14;
          sub_16160D0(a1 + 400, v22);
          v14 = v48;
          if ( (*(_BYTE *)(a1 + 408) & 1) != 0 )
          {
            v38 = a1 + 416;
            v39 = 7;
          }
          else
          {
            v37 = *(_DWORD *)(a1 + 424);
            v38 = *(_QWORD *)(a1 + 416);
            if ( !v37 )
            {
LABEL_82:
              *(_DWORD *)(a1 + 408) = (2 * (*(_DWORD *)(a1 + 408) >> 1) + 2) | *(_DWORD *)(a1 + 408) & 1;
              BUG();
            }
            v39 = v37 - 1;
          }
          v40 = v39 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          result = (_QWORD *)(v38 + 16LL * v40);
          v41 = *result;
          if ( v20 != *result )
          {
            v42 = 1;
            v43 = 0;
            while ( v41 != -4 )
            {
              if ( !v43 && v41 == -8 )
                v43 = result;
              v40 = v39 & (v42 + v40);
              result = (_QWORD *)(v38 + 16LL * v40);
              v41 = *result;
              if ( v20 == *result )
                goto LABEL_41;
              ++v42;
            }
            goto LABEL_48;
          }
          goto LABEL_41;
        }
LABEL_18:
        *(_DWORD *)(a1 + 408) = (2 * (v23 >> 1) + 2) | v23 & 1;
        if ( *result != -4 )
          --*(_DWORD *)(a1 + 412);
        v15 += 8;
        result[1] = 0;
        *result = v20;
        result[1] = a2;
        if ( v14 == v15 )
          return result;
      }
    }
  }
  return result;
}
