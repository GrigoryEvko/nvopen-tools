// Function: sub_1C0B8A0
// Address: 0x1c0b8a0
//
void __fastcall sub_1C0B8A0(__int64 a1, __int64 a2)
{
  int v3; // eax
  _QWORD *v4; // rdi
  unsigned int i; // esi
  _QWORD *v6; // rax
  _QWORD *v7; // r8
  unsigned int v8; // eax
  int v9; // r14d
  __int64 *v10; // rbx
  unsigned int v11; // ecx
  __int64 *v12; // r10
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // eax
  int v17; // eax
  unsigned __int64 v18; // rax
  __int64 v19; // rbx
  unsigned int v20; // r13d
  int v21; // r14d
  int v22; // r8d
  __int64 v23; // r9
  __int64 v24; // rax
  unsigned int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rax
  unsigned __int64 v28; // r12
  unsigned int v29; // eax
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  _QWORD *v32; // rax
  __int64 v33; // [rsp+10h] [rbp-A0h]
  __int64 v34; // [rsp+18h] [rbp-98h] BYREF
  __int64 *v35; // [rsp+28h] [rbp-88h] BYREF
  _QWORD *v36; // [rsp+30h] [rbp-80h] BYREF
  __int64 v37; // [rsp+38h] [rbp-78h]
  _QWORD v38[14]; // [rsp+40h] [rbp-70h] BYREF

  v3 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v34 = a1;
  v4 = *(_QWORD **)(a2 + 8);
  i = *(_DWORD *)(a2 + 24);
  if ( v3 )
  {
    v25 = 4 * v3;
    if ( (unsigned int)(4 * v3) < 0x40 )
      v25 = 64;
    if ( v25 >= i )
      goto LABEL_4;
    v26 = v3 - 1;
    if ( v26 )
    {
      _BitScanReverse(&v26, v26);
      v27 = (unsigned int)(1 << (33 - (v26 ^ 0x1F)));
      if ( (int)v27 < 64 )
        v27 = 64;
      if ( (_DWORD)v27 == i )
      {
        *(_QWORD *)(a2 + 16) = 0;
        v32 = &v4[v27];
        do
        {
          if ( v4 )
            *v4 = -8;
          ++v4;
        }
        while ( v32 != v4 );
        v4 = *(_QWORD **)(a2 + 8);
        i = *(_DWORD *)(a2 + 24);
        goto LABEL_8;
      }
      v28 = 4 * (int)v27 / 3u + 1;
    }
    else
    {
      v28 = 86;
    }
    j___libc_free_0(v4);
    v29 = sub_1454B60(v28);
    *(_DWORD *)(a2 + 24) = v29;
    if ( v29 )
    {
      v30 = (_QWORD *)sub_22077B0(8LL * v29);
      *(_QWORD *)(a2 + 16) = 0;
      v4 = v30;
      *(_QWORD *)(a2 + 8) = v30;
      v31 = &v30[*(unsigned int *)(a2 + 24)];
      for ( i = *(_DWORD *)(a2 + 24); v31 != v30; ++v30 )
      {
        if ( v30 )
          *v30 = -8;
      }
      goto LABEL_8;
    }
LABEL_53:
    *(_QWORD *)(a2 + 8) = 0;
    i = 0;
    v4 = 0;
    *(_QWORD *)(a2 + 16) = 0;
    goto LABEL_8;
  }
  if ( *(_DWORD *)(a2 + 20) )
  {
    if ( i <= 0x40 )
    {
LABEL_4:
      v6 = &v4[i];
      if ( v6 != v4 )
      {
        do
          *v4++ = -8;
        while ( v6 != v4 );
        v4 = *(_QWORD **)(a2 + 8);
        i = *(_DWORD *)(a2 + 24);
      }
      *(_QWORD *)(a2 + 16) = 0;
      goto LABEL_8;
    }
    j___libc_free_0(v4);
    *(_DWORD *)(a2 + 24) = 0;
    goto LABEL_53;
  }
LABEL_8:
  v36 = v38;
  v7 = v38;
  v38[0] = v34;
  v37 = 0x800000001LL;
  v8 = 1;
  while ( 1 )
  {
    v14 = v8--;
    v15 = v7[v14 - 1];
    LODWORD(v37) = v8;
    v34 = v15;
    if ( !i )
    {
      ++*(_QWORD *)a2;
LABEL_14:
      i *= 2;
LABEL_15:
      sub_13B3D40(a2, i);
      sub_1898220(a2, &v34, &v35);
      v10 = v35;
      v15 = v34;
      v16 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_25;
    }
    v9 = 1;
    v10 = 0;
    v11 = (i - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v12 = &v4[v11];
    v13 = *v12;
    if ( v15 != *v12 )
      break;
LABEL_10:
    if ( !v8 )
      goto LABEL_34;
LABEL_11:
    v4 = *(_QWORD **)(a2 + 8);
    i = *(_DWORD *)(a2 + 24);
  }
  while ( v13 != -8 )
  {
    if ( v13 != -16 || v10 )
      v12 = v10;
    v11 = (i - 1) & (v9 + v11);
    v13 = v4[v11];
    if ( v15 == v13 )
      goto LABEL_10;
    ++v9;
    v10 = v12;
    v12 = &v4[v11];
  }
  v17 = *(_DWORD *)(a2 + 16);
  if ( !v10 )
    v10 = v12;
  ++*(_QWORD *)a2;
  v16 = v17 + 1;
  if ( 4 * v16 >= 3 * i )
    goto LABEL_14;
  if ( i - (v16 + *(_DWORD *)(a2 + 20)) <= i >> 3 )
    goto LABEL_15;
LABEL_25:
  *(_DWORD *)(a2 + 16) = v16;
  if ( *v10 != -8 )
    --*(_DWORD *)(a2 + 20);
  *v10 = v15;
  v18 = sub_157EBA0(v34);
  v19 = v18;
  if ( v18 && (v20 = 0, (v21 = sub_15F4D60(v18)) != 0) )
  {
    do
    {
      v23 = sub_15F4DF0(v19, v20);
      v24 = (unsigned int)v37;
      if ( (unsigned int)v37 >= HIDWORD(v37) )
      {
        v33 = v23;
        sub_16CD150((__int64)&v36, v38, 0, 8, v22, v23);
        v24 = (unsigned int)v37;
        v23 = v33;
      }
      ++v20;
      v36[v24] = v23;
      v8 = v37 + 1;
      LODWORD(v37) = v37 + 1;
    }
    while ( v21 != v20 );
    v7 = v36;
  }
  else
  {
    v8 = v37;
    v7 = v36;
  }
  if ( v8 )
    goto LABEL_11;
LABEL_34:
  if ( v7 != v38 )
    _libc_free((unsigned __int64)v7);
}
