// Function: sub_210AA20
// Address: 0x210aa20
//
__int64 __fastcall sub_210AA20(__int64 *a1, __int64 a2)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // r9
  unsigned int v10; // r13d
  int v12; // r11d
  __int64 *v13; // rdx
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rax
  unsigned int v17; // eax
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r12
  unsigned __int64 v23; // rdi
  int v24; // eax
  int v25; // esi
  __int64 v26; // r8
  unsigned int v27; // eax
  __int64 v28; // rdi
  int v29; // r10d
  __int64 *v30; // r9
  int v31; // eax
  int v32; // eax
  __int64 v33; // rdi
  int v34; // r9d
  unsigned int v35; // r13d
  __int64 *v36; // r8
  __int64 v37; // rsi
  __int64 v38[4]; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v39; // [rsp+20h] [rbp-B0h]
  __int64 v40; // [rsp+28h] [rbp-A8h]
  int v41; // [rsp+30h] [rbp-A0h]
  __int64 v42; // [rsp+38h] [rbp-98h]
  __int64 v43; // [rsp+40h] [rbp-90h]
  unsigned __int64 *v44; // [rsp+48h] [rbp-88h]
  __int64 v45; // [rsp+50h] [rbp-80h]
  _BYTE v46[32]; // [rsp+58h] [rbp-78h] BYREF
  unsigned __int64 *v47; // [rsp+78h] [rbp-58h]
  __int64 v48; // [rsp+80h] [rbp-50h]
  _QWORD v49[9]; // [rsp+88h] [rbp-48h] BYREF

  v4 = *a1;
  v5 = *(_DWORD *)(*a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)v4;
    goto LABEL_25;
  }
  v6 = *(_QWORD *)(v4 + 8);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 == a2 )
  {
LABEL_3:
    v10 = *((_DWORD *)v8 + 2);
    if ( v10 )
      return v10;
    goto LABEL_14;
  }
  v12 = 1;
  v13 = 0;
  while ( v9 != -8 )
  {
    if ( !v13 && v9 == -16 )
      v13 = v8;
    v7 = (v5 - 1) & (v12 + v7);
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
      goto LABEL_3;
    ++v12;
  }
  if ( !v13 )
    v13 = v8;
  v14 = *(_DWORD *)(v4 + 16);
  ++*(_QWORD *)v4;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_25:
    sub_1DA35E0(v4, 2 * v5);
    v24 = *(_DWORD *)(v4 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(v4 + 8);
      v27 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(v4 + 16) + 1;
      v13 = (__int64 *)(v26 + 16LL * v27);
      v28 = *v13;
      if ( *v13 != a2 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -8 )
        {
          if ( !v30 && v28 == -16 )
            v30 = v13;
          v27 = v25 & (v29 + v27);
          v13 = (__int64 *)(v26 + 16LL * v27);
          v28 = *v13;
          if ( *v13 == a2 )
            goto LABEL_11;
          ++v29;
        }
        if ( v30 )
          v13 = v30;
      }
      goto LABEL_11;
    }
    goto LABEL_53;
  }
  if ( v5 - *(_DWORD *)(v4 + 20) - v15 <= v5 >> 3 )
  {
    sub_1DA35E0(v4, v5);
    v31 = *(_DWORD *)(v4 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(v4 + 8);
      v34 = 1;
      v35 = v32 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v36 = 0;
      v15 = *(_DWORD *)(v4 + 16) + 1;
      v13 = (__int64 *)(v33 + 16LL * v35);
      v37 = *v13;
      if ( *v13 != a2 )
      {
        while ( v37 != -8 )
        {
          if ( !v36 && v37 == -16 )
            v36 = v13;
          v35 = v32 & (v34 + v35);
          v13 = (__int64 *)(v33 + 16LL * v35);
          v37 = *v13;
          if ( *v13 == a2 )
            goto LABEL_11;
          ++v34;
        }
        if ( v36 )
          v13 = v36;
      }
      goto LABEL_11;
    }
LABEL_53:
    ++*(_DWORD *)(v4 + 16);
    BUG();
  }
LABEL_11:
  *(_DWORD *)(v4 + 16) = v15;
  if ( *v13 != -8 )
    --*(_DWORD *)(v4 + 20);
  *v13 = a2;
  *((_DWORD *)v13 + 2) = 0;
LABEL_14:
  v16 = a1[3];
  v38[0] = (__int64)a1;
  v38[1] = v4;
  v38[2] = v16;
  v38[3] = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = (unsigned __int64 *)v46;
  v45 = 0x400000000LL;
  v47 = v49;
  v48 = 0;
  v49[0] = 0;
  v49[1] = 1;
  v17 = sub_2109FF0(v38, a2);
  v18 = v44;
  v10 = v17;
  v19 = &v44[(unsigned int)v45];
  if ( v44 != v19 )
  {
    do
    {
      v20 = *v18++;
      _libc_free(v20);
    }
    while ( v19 != v18 );
  }
  v21 = v47;
  v22 = &v47[2 * (unsigned int)v48];
  if ( v47 != v22 )
  {
    do
    {
      v23 = *v21;
      v21 += 2;
      _libc_free(v23);
    }
    while ( v21 != v22 );
    v22 = v47;
  }
  if ( v22 != v49 )
    _libc_free((unsigned __int64)v22);
  if ( v44 != (unsigned __int64 *)v46 )
    _libc_free((unsigned __int64)v44);
  j___libc_free_0(v39);
  return v10;
}
