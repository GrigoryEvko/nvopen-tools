// Function: sub_2337550
// Address: 0x2337550
//
void __fastcall sub_2337550(__int64 *a1)
{
  __int64 v1; // r13
  int v2; // eax
  unsigned int v3; // ecx
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  int v7; // r15d
  unsigned int v8; // eax
  _QWORD **v9; // rbx
  __int64 v10; // rdx
  _QWORD **v11; // r14
  _QWORD **v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // rbx
  unsigned __int64 v15; // r15
  __int64 v16; // rdi
  int v17; // edx
  _QWORD **k; // rbx
  __int64 v19; // rax
  _QWORD *v20; // r12
  unsigned __int64 v21; // r8
  __int64 v22; // rdi
  int v23; // r14d
  unsigned int v24; // r15d
  unsigned int v25; // eax
  _QWORD *v26; // rdi
  unsigned int v27; // eax
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *m; // rdx
  unsigned int v31; // eax
  _QWORD *v32; // rdi
  __int64 v33; // r12
  _QWORD *v34; // rax
  unsigned int v35; // eax
  _QWORD *v36; // rax
  __int64 v37; // rdx
  _QWORD *j; // rdx
  _QWORD *v39; // rax
  __int64 v40; // [rsp+0h] [rbp-40h]
  unsigned __int64 v41; // [rsp+8h] [rbp-38h]

  v1 = *a1;
  if ( !*a1 )
    return;
  v2 = *(_DWORD *)(v1 + 80);
  ++*(_QWORD *)(v1 + 64);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(v1 + 84) )
      goto LABEL_9;
    v4 = *(unsigned int *)(v1 + 88);
    if ( (unsigned int)v4 <= 0x40 )
      goto LABEL_6;
    sub_C7D6A0(*(_QWORD *)(v1 + 72), 24 * v4, 8);
    *(_DWORD *)(v1 + 88) = 0;
    goto LABEL_33;
  }
  v3 = 4 * v2;
  v4 = *(unsigned int *)(v1 + 88);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v3 = 64;
  if ( (unsigned int)v4 <= v3 )
  {
LABEL_6:
    v5 = *(_QWORD **)(v1 + 72);
    for ( i = &v5[3 * v4]; i != v5; *(v5 - 2) = -4096 )
    {
      *v5 = -4096;
      v5 += 3;
    }
    goto LABEL_8;
  }
  v31 = v2 - 1;
  if ( v31 )
  {
    _BitScanReverse(&v31, v31);
    v32 = *(_QWORD **)(v1 + 72);
    v33 = (unsigned int)(1 << (33 - (v31 ^ 0x1F)));
    if ( (int)v33 < 64 )
      v33 = 64;
    if ( (_DWORD)v33 == (_DWORD)v4 )
    {
      *(_QWORD *)(v1 + 80) = 0;
      v34 = &v32[3 * v33];
      do
      {
        if ( v32 )
        {
          *v32 = -4096;
          v32[1] = -4096;
        }
        v32 += 3;
      }
      while ( v34 != v32 );
      goto LABEL_9;
    }
  }
  else
  {
    v32 = *(_QWORD **)(v1 + 72);
    LODWORD(v33) = 64;
  }
  sub_C7D6A0((__int64)v32, 24 * v4, 8);
  v35 = sub_2309150(v33);
  *(_DWORD *)(v1 + 88) = v35;
  if ( !v35 )
  {
LABEL_33:
    *(_QWORD *)(v1 + 72) = 0;
LABEL_8:
    *(_QWORD *)(v1 + 80) = 0;
    goto LABEL_9;
  }
  v36 = (_QWORD *)sub_C7D670(24LL * v35, 8);
  v37 = *(unsigned int *)(v1 + 88);
  *(_QWORD *)(v1 + 80) = 0;
  *(_QWORD *)(v1 + 72) = v36;
  for ( j = &v36[3 * v37]; j != v36; v36 += 3 )
  {
    if ( v36 )
    {
      *v36 = -4096;
      v36[1] = -4096;
    }
  }
LABEL_9:
  v7 = *(_DWORD *)(v1 + 48);
  ++*(_QWORD *)(v1 + 32);
  if ( v7 || *(_DWORD *)(v1 + 52) )
  {
    v8 = 4 * v7;
    v9 = *(_QWORD ***)(v1 + 40);
    v10 = *(unsigned int *)(v1 + 56);
    if ( (unsigned int)(4 * v7) < 0x40 )
      v8 = 64;
    v40 = 32 * v10;
    v11 = &v9[4 * v10];
    if ( (unsigned int)v10 <= v8 )
    {
      v12 = v9 + 1;
      if ( v9 != v11 )
      {
        while ( 1 )
        {
          v13 = (__int64)*(v12 - 1);
          if ( v13 != -4096 )
          {
            if ( v13 != -8192 )
            {
              v14 = *v12;
              while ( v14 != v12 )
              {
                v15 = (unsigned __int64)v14;
                v14 = (_QWORD *)*v14;
                v16 = *(_QWORD *)(v15 + 24);
                if ( v16 )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
                j_j___libc_free_0(v15);
              }
            }
            *(v12 - 1) = (_QWORD *)-4096LL;
          }
          if ( v11 == v12 + 3 )
            break;
          v12 += 4;
        }
      }
      goto LABEL_28;
    }
    for ( k = v9 + 1; ; k += 4 )
    {
      v19 = (__int64)*(k - 1);
      if ( v19 != -8192 && v19 != -4096 )
      {
        v20 = *k;
        while ( k != v20 )
        {
          v21 = (unsigned __int64)v20;
          v20 = (_QWORD *)*v20;
          v22 = *(_QWORD *)(v21 + 24);
          if ( v22 )
          {
            v41 = v21;
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v22 + 8LL))(v22);
            v21 = v41;
          }
          j_j___libc_free_0(v21);
        }
      }
      if ( v11 == k + 3 )
        break;
    }
    v17 = *(_DWORD *)(v1 + 56);
    if ( !v7 )
    {
      if ( !v17 )
      {
LABEL_28:
        *(_QWORD *)(v1 + 48) = 0;
        return;
      }
      sub_C7D6A0(*(_QWORD *)(v1 + 40), v40, 8);
      *(_DWORD *)(v1 + 56) = 0;
LABEL_27:
      *(_QWORD *)(v1 + 40) = 0;
      goto LABEL_28;
    }
    v23 = 64;
    v24 = v7 - 1;
    if ( v24 )
    {
      _BitScanReverse(&v25, v24);
      v23 = 1 << (33 - (v25 ^ 0x1F));
      if ( v23 < 64 )
        v23 = 64;
    }
    v26 = *(_QWORD **)(v1 + 40);
    if ( v23 == v17 )
    {
      *(_QWORD *)(v1 + 48) = 0;
      v39 = &v26[4 * (unsigned int)v23];
      do
      {
        if ( v26 )
          *v26 = -4096;
        v26 += 4;
      }
      while ( v39 != v26 );
    }
    else
    {
      sub_C7D6A0((__int64)v26, v40, 8);
      v27 = sub_2309150(v23);
      *(_DWORD *)(v1 + 56) = v27;
      if ( !v27 )
        goto LABEL_27;
      v28 = (_QWORD *)sub_C7D670(32LL * v27, 8);
      v29 = *(unsigned int *)(v1 + 56);
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 40) = v28;
      for ( m = &v28[4 * v29]; m != v28; v28 += 4 )
      {
        if ( v28 )
          *v28 = -4096;
      }
    }
  }
}
