// Function: sub_146DBF0
// Address: 0x146dbf0
//
void __fastcall sub_146DBF0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  _BYTE *v4; // rdx
  __int64 v5; // r10
  _BYTE *v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // r12
  _QWORD *v9; // rbx
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rax
  _QWORD *v17; // rax
  int v18; // ecx
  int v19; // ecx
  __int64 v20; // r8
  unsigned int v21; // eax
  int v22; // edx
  __int64 v23; // rsi
  int v24; // r9d
  __int64 *v25; // rdi
  __int64 *v26; // r9
  int v27; // edi
  int v28; // ecx
  int v29; // ecx
  __int64 v30; // r8
  int v31; // r9d
  unsigned int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // [rsp+8h] [rbp-B8h]
  __int64 v35; // [rsp+10h] [rbp-B0h]
  __int64 v36; // [rsp+10h] [rbp-B0h]
  int v37; // [rsp+10h] [rbp-B0h]
  unsigned int v38; // [rsp+10h] [rbp-B0h]
  __int64 v39; // [rsp+18h] [rbp-A8h]
  __int64 v40; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE *v41; // [rsp+28h] [rbp-98h]
  _BYTE *v42; // [rsp+30h] [rbp-90h]
  __int64 v43; // [rsp+38h] [rbp-88h]
  int v44; // [rsp+40h] [rbp-80h]
  _BYTE v45[120]; // [rsp+48h] [rbp-78h] BYREF

  v41 = v45;
  v40 = 0;
  v42 = v45;
  v43 = 8;
  v44 = 0;
  sub_145B270(a1, a2, (__int64)&v40);
  v3 = (unsigned __int64)v42;
  v4 = v41;
  v5 = a2;
  if ( v42 == v41 )
    v6 = &v42[8 * HIDWORD(v43)];
  else
    v6 = &v42[8 * (unsigned int)v43];
  if ( v42 == v6 )
    goto LABEL_7;
  v7 = v42;
  while ( 1 )
  {
    v8 = *v7;
    v9 = v7;
    if ( *v7 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v6 == (_BYTE *)++v7 )
      goto LABEL_7;
  }
  if ( v6 != (_BYTE *)v7 )
  {
    v10 = *(_DWORD *)(a1 + 992);
    v39 = a1 + 968;
    if ( !v10 )
      goto LABEL_21;
LABEL_12:
    v11 = *(_QWORD *)(a1 + 976);
    v12 = (v10 - 1) & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
    v13 = (__int64 *)(v11 + 56LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_13;
    v37 = 1;
    v26 = 0;
    while ( v14 != -8 )
    {
      if ( v14 == -16 && !v26 )
        v26 = v13;
      v12 = (v10 - 1) & (v37 + v12);
      v13 = (__int64 *)(v11 + 56LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
      {
LABEL_13:
        v15 = *((unsigned int *)v13 + 4);
        if ( (unsigned int)v15 >= *((_DWORD *)v13 + 5) )
        {
          v36 = v5;
          sub_16CD150(v13 + 1, v13 + 3, 0, 8);
          v5 = v36;
          v16 = (__int64 *)(v13[1] + 8LL * *((unsigned int *)v13 + 4));
        }
        else
        {
          v16 = (__int64 *)(v13[1] + 8 * v15);
        }
        goto LABEL_15;
      }
      ++v37;
    }
    v27 = *(_DWORD *)(a1 + 984);
    if ( v26 )
      v13 = v26;
    ++*(_QWORD *)(a1 + 968);
    v22 = v27 + 1;
    if ( 4 * (v27 + 1) >= 3 * v10 )
    {
      while ( 1 )
      {
        v35 = v5;
        sub_146D9A0(v39, 2 * v10);
        v18 = *(_DWORD *)(a1 + 992);
        if ( !v18 )
          goto LABEL_58;
        v19 = v18 - 1;
        v20 = *(_QWORD *)(a1 + 976);
        v5 = v35;
        v21 = v19 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v13 = (__int64 *)(v20 + 56LL * v21);
        v22 = *(_DWORD *)(a1 + 984) + 1;
        v23 = *v13;
        if ( *v13 != v8 )
          break;
LABEL_36:
        *(_DWORD *)(a1 + 984) = v22;
        if ( *v13 != -8 )
          --*(_DWORD *)(a1 + 988);
        v16 = v13 + 3;
        *v13 = v8;
        v13[1] = (__int64)(v13 + 3);
        v13[2] = 0x400000000LL;
LABEL_15:
        *v16 = v5;
        v17 = v9 + 1;
        ++*((_DWORD *)v13 + 4);
        if ( v9 + 1 == (_QWORD *)v6 )
          goto LABEL_18;
        while ( 1 )
        {
          v8 = *v17;
          v9 = v17;
          if ( *v17 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v6 == (_BYTE *)++v17 )
            goto LABEL_18;
        }
        if ( v6 == (_BYTE *)v17 )
        {
LABEL_18:
          v3 = (unsigned __int64)v42;
          v4 = v41;
          goto LABEL_7;
        }
        v10 = *(_DWORD *)(a1 + 992);
        if ( v10 )
          goto LABEL_12;
LABEL_21:
        ++*(_QWORD *)(a1 + 968);
      }
      v24 = 1;
      v25 = 0;
      while ( v23 != -8 )
      {
        if ( !v25 && v23 == -16 )
          v25 = v13;
        v21 = v19 & (v24 + v21);
        v13 = (__int64 *)(v20 + 56LL * v21);
        v23 = *v13;
        if ( v8 == *v13 )
          goto LABEL_36;
        ++v24;
      }
    }
    else
    {
      if ( v10 - *(_DWORD *)(a1 + 988) - v22 > v10 >> 3 )
        goto LABEL_36;
      v34 = v5;
      v38 = ((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9);
      sub_146D9A0(v39, v10);
      v28 = *(_DWORD *)(a1 + 992);
      if ( !v28 )
      {
LABEL_58:
        ++*(_DWORD *)(a1 + 984);
        BUG();
      }
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 976);
      v31 = 1;
      v5 = v34;
      v32 = v29 & v38;
      v13 = (__int64 *)(v30 + 56LL * (v29 & v38));
      v22 = *(_DWORD *)(a1 + 984) + 1;
      v25 = 0;
      v33 = *v13;
      if ( *v13 == v8 )
        goto LABEL_36;
      while ( v33 != -8 )
      {
        if ( !v25 && v33 == -16 )
          v25 = v13;
        v32 = v29 & (v31 + v32);
        v13 = (__int64 *)(v30 + 56LL * v32);
        v33 = *v13;
        if ( v8 == *v13 )
          goto LABEL_36;
        ++v31;
      }
    }
    if ( v25 )
      v13 = v25;
    goto LABEL_36;
  }
LABEL_7:
  if ( v4 != (_BYTE *)v3 )
    _libc_free(v3);
}
