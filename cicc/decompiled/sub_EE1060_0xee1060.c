// Function: sub_EE1060
// Address: 0xee1060
//
__int64 __fastcall sub_EE1060(__int64 a1, unsigned int a2)
{
  unsigned int *v3; // r12
  __int64 result; // rax
  int *v5; // r12
  __int64 v6; // r9
  int v7; // r11d
  _QWORD *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  __int64 v14; // rdi
  int v15; // esi
  __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  __int64 *v18; // rdx
  _QWORD *v19; // r13
  unsigned int v20; // esi
  __int64 v21; // r14
  __int64 v22; // r15
  unsigned int v23; // r13d
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // rdi
  __int64 v27; // rsi
  int v28; // eax
  int v29; // r11d
  _QWORD *v30; // r10
  int v31; // eax
  unsigned __int64 v32; // r8
  __int64 v33; // r15
  __int64 *v34; // rdx
  _QWORD *v35; // r12
  int v36; // esi
  int v37; // esi
  __int64 v38; // rdi
  int v39; // r11d
  __int64 v40; // rcx
  __int64 v41; // [rsp+10h] [rbp-70h]
  int v42; // [rsp+1Ch] [rbp-64h]
  __int64 v43; // [rsp+28h] [rbp-58h]
  __int64 v44; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v45; // [rsp+38h] [rbp-48h]
  __int64 v46; // [rsp+40h] [rbp-40h]

  v41 = a1 + 24;
  v43 = 0;
  v3 = (unsigned int *)(*(_QWORD *)a1 + 4LL * a2);
  result = *v3;
  v5 = (int *)(v3 + 1);
  v42 = result;
  if ( !(_DWORD)result )
    return result;
  while ( 1 )
  {
    if ( *v5 < 0 )
      v5 += (unsigned int)-*v5;
    (*(void (__fastcall **)(__int64 *, _QWORD))(a1 + 8))(&v44, *(_QWORD *)(a1 + 16));
    v20 = *(_DWORD *)(a1 + 48);
    v21 = v44;
    v22 = HIDWORD(v46);
    v23 = v46;
    if ( !v20 )
    {
      ++*(_QWORD *)(a1 + 24);
      goto LABEL_19;
    }
    v6 = *(_QWORD *)(a1 + 32);
    v7 = 1;
    v8 = 0;
    v9 = ((unsigned int)((0xBF58476D1CE4E5B9LL * v44) >> 31) ^ (484763065 * (_DWORD)v44)) & (v20 - 1);
    v10 = v6 + 24 * v9;
    v11 = *(_QWORD *)v10;
    if ( v44 != *(_QWORD *)v10 )
    {
      while ( v11 != -1 )
      {
        if ( !v8 && v11 == -2 )
          v8 = (_QWORD *)v10;
        LODWORD(v9) = (v20 - 1) & (v7 + v9);
        v10 = v6 + 24LL * (unsigned int)v9;
        v11 = *(_QWORD *)v10;
        if ( v44 == *(_QWORD *)v10 )
          goto LABEL_4;
        ++v7;
      }
      if ( !v8 )
        v8 = (_QWORD *)v10;
      v31 = *(_DWORD *)(a1 + 40);
      ++*(_QWORD *)(a1 + 24);
      v28 = v31 + 1;
      if ( 4 * v28 < 3 * v20 )
      {
        if ( v20 - *(_DWORD *)(a1 + 44) - v28 > v20 >> 3 )
          goto LABEL_36;
        sub_EE0D70(v41, v20);
        v36 = *(_DWORD *)(a1 + 48);
        if ( !v36 )
        {
LABEL_61:
          ++*(_DWORD *)(a1 + 40);
          BUG();
        }
        v37 = v36 - 1;
        v38 = *(_QWORD *)(a1 + 32);
        v30 = 0;
        v39 = 1;
        v6 = v37 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v21) >> 31) ^ (484763065 * (_DWORD)v21));
        v8 = (_QWORD *)(v38 + 24 * v6);
        v40 = *v8;
        v28 = *(_DWORD *)(a1 + 40) + 1;
        if ( v21 == *v8 )
          goto LABEL_36;
        while ( v40 != -1 )
        {
          if ( !v30 && v40 == -2 )
            v30 = v8;
          v6 = v37 & (unsigned int)(v6 + v39);
          v8 = (_QWORD *)(v38 + 24 * v6);
          v40 = *v8;
          if ( v21 == *v8 )
            goto LABEL_36;
          ++v39;
        }
        goto LABEL_23;
      }
LABEL_19:
      sub_EE0D70(v41, 2 * v20);
      v24 = *(_DWORD *)(a1 + 48);
      if ( !v24 )
        goto LABEL_61;
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 32);
      v6 = v25 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v21) >> 31) ^ (484763065 * (_DWORD)v21));
      v8 = (_QWORD *)(v26 + 24 * v6);
      v27 = *v8;
      v28 = *(_DWORD *)(a1 + 40) + 1;
      if ( v21 == *v8 )
        goto LABEL_36;
      v29 = 1;
      v30 = 0;
      while ( v27 != -1 )
      {
        if ( !v30 && v27 == -2 )
          v30 = v8;
        v6 = v25 & (unsigned int)(v6 + v29);
        v8 = (_QWORD *)(v26 + 24 * v6);
        v27 = *v8;
        if ( v21 == *v8 )
          goto LABEL_36;
        ++v29;
      }
LABEL_23:
      if ( v30 )
        v8 = v30;
LABEL_36:
      *(_DWORD *)(a1 + 40) = v28;
      if ( *v8 != -1 )
        --*(_DWORD *)(a1 + 44);
      *v8 = v21;
      v14 = (__int64)(v8 + 1);
      v32 = 1;
      v8[1] = v8 + 3;
      v8[2] = 0;
      v33 = (v22 << 32) | v23;
LABEL_39:
      sub_C8D5F0(v14, (const void *)(v14 + 16), v32, 0x10u, v32, v6);
      v12 = *(unsigned int *)(v14 + 8);
LABEL_40:
      v34 = (__int64 *)(*(_QWORD *)v14 + 16 * v12);
      *v34 = v33;
      v34[1] = v43;
      ++*(_DWORD *)(v14 + 8);
      goto LABEL_8;
    }
LABEL_4:
    v12 = *(unsigned int *)(v10 + 16);
    v13 = *(unsigned int *)(v10 + 20);
    v14 = v10 + 8;
    v15 = *(_DWORD *)(v10 + 16);
    if ( v13 <= v12 )
    {
      v32 = v12 + 1;
      v33 = v46;
      if ( v12 + 1 <= v13 )
        goto LABEL_40;
      goto LABEL_39;
    }
    v16 = *(_QWORD *)(v10 + 8) + 16 * v12;
    if ( v16 )
    {
      *(_DWORD *)v16 = v46;
      *(_DWORD *)(v16 + 4) = v22;
      *(_QWORD *)(v16 + 8) = v43;
      v15 = *(_DWORD *)(v10 + 16);
    }
    *(_DWORD *)(v10 + 16) = v15 + 1;
LABEL_8:
    v17 = ((unsigned __int64)v5 - *(_QWORD *)a1) >> 2;
    result = 1LL << v17;
    v18 = (__int64 *)(*(_QWORD *)(a1 + 56) + 8LL * ((unsigned int)v17 >> 6));
    if ( (*v18 & (1LL << v17)) != 0 )
      break;
    result |= *v18;
    ++v5;
    *v18 = result;
    v19 = v45;
    if ( v45 )
    {
      if ( (_QWORD *)*v45 != v45 + 2 )
        j_j___libc_free_0(*v45, v45[2] + 1LL);
      result = j_j___libc_free_0(v19, 32);
    }
    if ( !--v42 )
      return result;
    v43 = v21;
  }
  v35 = v45;
  if ( v45 )
  {
    if ( (_QWORD *)*v45 != v45 + 2 )
      j_j___libc_free_0(*v45, v45[2] + 1LL);
    return j_j___libc_free_0(v35, 32);
  }
  return result;
}
