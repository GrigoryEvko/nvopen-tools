// Function: sub_1EE80E0
// Address: 0x1ee80e0
//
__int64 __fastcall sub_1EE80E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // r8
  int v8; // r9d
  __int64 v9; // rdi
  __int64 v10; // r13
  int v11; // r15d
  unsigned int v12; // esi
  _DWORD *v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 v16; // rdi
  int v17; // r12d
  __int64 v18; // rcx
  __int64 v19; // r10
  unsigned int *v20; // r14
  unsigned int *v21; // r12
  unsigned int v22; // ecx
  __int64 v23; // r9
  unsigned int v24; // esi
  unsigned __int64 v25; // r13
  _BYTE *v26; // r8
  unsigned int v27; // eax
  __int64 v28; // rdi
  _DWORD *v29; // rdx
  int v30; // r8d
  unsigned int v31; // ecx
  int v32; // esi
  __int64 v33; // rcx
  __int64 result; // rax
  __int64 v35; // r14
  unsigned int v36; // ecx
  unsigned int v37; // esi
  unsigned __int64 v38; // r14
  _BYTE *v39; // r11
  unsigned int v40; // eax
  __int64 v41; // rdi
  _DWORD *v42; // rdx
  int v43; // ecx
  unsigned int v44; // r10d
  __int64 v45; // rsi
  int v46; // ecx
  unsigned int v47; // eax
  __int64 v48; // rdi
  _DWORD *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned __int64 v53; // [rsp+8h] [rbp-48h]
  int v54; // [rsp+10h] [rbp-40h]
  int v55; // [rsp+14h] [rbp-3Ch]
  __int64 v56; // [rsp+18h] [rbp-38h]

  if ( !sub_1EE61D0(a1) )
    sub_1EE6350(a1, a2, v3, v4, v5, v6);
  v53 = 0;
  if ( *(_BYTE *)(a1 + 56) )
    v53 = sub_1EE6230((_QWORD *)a1);
  if ( sub_1EE6200(a1) )
  {
    v9 = *(_QWORD *)(a1 + 48);
    if ( *(_BYTE *)(a1 + 56) )
      sub_1EE6000(v9, v53);
    else
      sub_1EE6040(v9, *(_QWORD *)(a1 + 64));
  }
  v10 = *(_QWORD *)a2;
  v56 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v56 )
  {
    do
    {
      v11 = *(_DWORD *)v10;
      v12 = *(_DWORD *)v10;
      if ( *(int *)v10 < 0 )
        v12 = *(_DWORD *)(a1 + 192) + (v12 & 0x7FFFFFFF);
      v13 = *(_DWORD **)(a1 + 176);
      v14 = *(unsigned int *)(a1 + 104);
      v15 = *((unsigned __int8 *)v13 + v12);
      if ( v15 >= (unsigned int)v14 )
        goto LABEL_38;
      v16 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v13 = (_DWORD *)(v16 + 8LL * v15);
        if ( v12 == *v13 )
          break;
        v15 += 256;
        if ( (unsigned int)v14 <= v15 )
          goto LABEL_38;
      }
      if ( v13 == (_DWORD *)(v16 + 8 * v14) )
      {
LABEL_38:
        v18 = *(unsigned int *)(v10 + 4);
        v17 = 0;
        v19 = v18;
        if ( !*(_DWORD *)(v10 + 4) )
          goto LABEL_17;
      }
      else
      {
        v17 = v13[1];
        v18 = *(unsigned int *)(v10 + 4);
        v19 = (unsigned int)v18 & ~v17;
        if ( ((unsigned int)v18 & ~v17) == 0 )
          goto LABEL_17;
      }
      v55 = v19;
      v35 = (unsigned int)v19;
      v54 = v18;
      sub_1EE7560(a1, (v19 << 32) | (unsigned int)v11, (__int64)v13, v18, v7, v8);
      sub_1EE5D10(a1, v11, v17, v17 | v54);
      v36 = v11;
      if ( v11 < 0 )
        v36 = *(_DWORD *)(a1 + 192) + (v11 & 0x7FFFFFFF);
      v37 = *(_DWORD *)(a1 + 104);
      v38 = v36 | (unsigned __int64)(v35 << 32);
      v39 = (_BYTE *)(*(_QWORD *)(a1 + 176) + v36);
      v40 = (unsigned __int8)*v39;
      if ( v40 >= v37 )
        goto LABEL_64;
      v41 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v42 = (_DWORD *)(v41 + 8LL * v40);
        if ( v36 == *v42 )
          break;
        v40 += 256;
        if ( v37 <= v40 )
          goto LABEL_64;
      }
      if ( v42 == (_DWORD *)(v41 + 8LL * v37) )
      {
LABEL_64:
        *v39 = v37;
        v51 = *(unsigned int *)(a1 + 104);
        if ( (unsigned int)v51 >= *(_DWORD *)(a1 + 108) )
        {
          sub_16CD150(a1 + 96, (const void *)(a1 + 112), 0, 8, v7, v8);
          v51 = *(unsigned int *)(a1 + 104);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v51) = v38;
        ++*(_DWORD *)(a1 + 104);
LABEL_17:
        if ( *(_BYTE *)(a1 + 56) )
          goto LABEL_47;
        goto LABEL_18;
      }
      v42[1] |= v55;
      if ( *(_BYTE *)(a1 + 56) )
      {
LABEL_47:
        v43 = sub_1EE80B0(a1, v11, v53);
        if ( v43 )
        {
          v44 = v11;
          if ( v11 < 0 )
            v44 = *(_DWORD *)(a1 + 192) + (v11 & 0x7FFFFFFF);
          v45 = *(unsigned int *)(a1 + 104);
          v46 = ~v43;
          v47 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 176) + v44);
          if ( v47 < (unsigned int)v45 )
          {
            v48 = *(_QWORD *)(a1 + 96);
            while ( 1 )
            {
              v49 = (_DWORD *)(v48 + 8LL * v47);
              if ( v44 == *v49 )
                break;
              v47 += 256;
              if ( (unsigned int)v45 <= v47 )
                goto LABEL_56;
            }
            if ( v49 != (_DWORD *)(v48 + 8 * v45) )
              v49[1] &= v46;
          }
LABEL_56:
          sub_1EE5E20(a1, v11, v17, v17 & v46);
        }
      }
LABEL_18:
      v10 += 8;
    }
    while ( v56 != v10 );
  }
  v20 = *(unsigned int **)(a2 + 80);
  v21 = &v20[2 * *(unsigned int *)(a2 + 88)];
  while ( v21 != v20 )
  {
    v22 = *v20;
    v23 = v20[1];
    if ( (*v20 & 0x80000000) != 0 )
      v22 = *(_DWORD *)(a1 + 192) + (v22 & 0x7FFFFFFF);
    v24 = *(_DWORD *)(a1 + 104);
    v25 = v22 | (unsigned __int64)(v23 << 32);
    v26 = (_BYTE *)(*(_QWORD *)(a1 + 176) + v22);
    v27 = (unsigned __int8)*v26;
    if ( v27 >= v24 )
      goto LABEL_57;
    v28 = *(_QWORD *)(a1 + 96);
    while ( 1 )
    {
      v29 = (_DWORD *)(v28 + 8LL * v27);
      if ( v22 == *v29 )
        break;
      v27 += 256;
      if ( v24 <= v27 )
        goto LABEL_57;
    }
    if ( v29 == (_DWORD *)(v28 + 8LL * v24) )
    {
LABEL_57:
      *v26 = v24;
      v50 = *(unsigned int *)(a1 + 104);
      if ( (unsigned int)v50 >= *(_DWORD *)(a1 + 108) )
      {
        sub_16CD150(a1 + 96, (const void *)(a1 + 112), 0, 8, (int)v26, v23);
        v50 = *(unsigned int *)(a1 + 104);
      }
      v30 = 0;
      *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v50) = v25;
      ++*(_DWORD *)(a1 + 104);
    }
    else
    {
      v30 = v29[1];
      v29[1] = v30 | v23;
    }
    v31 = v20[1];
    v32 = *v20;
    v20 += 2;
    sub_1EE5D10(a1, v32, v30, v30 | v31);
  }
  sub_1EE7580(a1, *(int **)(a2 + 160), *(unsigned int *)(a2 + 168));
  v33 = *(_QWORD *)(a1 + 40) + 24LL;
  result = *(_QWORD *)(a1 + 64);
  if ( !result )
    BUG();
  if ( (*(_BYTE *)result & 4) == 0 && (*(_BYTE *)(result + 46) & 8) != 0 )
  {
    do
      result = *(_QWORD *)(result + 8);
    while ( (*(_BYTE *)(result + 46) & 8) != 0 );
  }
LABEL_31:
  for ( result = *(_QWORD *)(result + 8); v33 != result; result = *(_QWORD *)(result + 8) )
  {
    if ( (unsigned __int16)(**(_WORD **)(result + 16) - 12) > 1u )
      break;
    if ( (*(_BYTE *)result & 4) != 0 || (*(_BYTE *)(result + 46) & 8) == 0 )
      goto LABEL_31;
    do
      result = *(_QWORD *)(result + 8);
    while ( (*(_BYTE *)(result + 46) & 8) != 0 );
  }
  *(_QWORD *)(a1 + 64) = result;
  return result;
}
