// Function: sub_1D8E690
// Address: 0x1d8e690
//
__int64 __fastcall sub_1D8E690(__int64 a1)
{
  _QWORD **v2; // r15
  _QWORD **v3; // r14
  _QWORD **v4; // rbx
  __int64 v5; // r12
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *j; // rdx
  __int64 v10; // r14
  __int64 result; // rax
  __int64 v12; // rbx
  _QWORD *v13; // r12
  __int64 (__fastcall *v14)(_QWORD *); // rax
  _QWORD *v15; // rdi
  unsigned int v16; // ecx
  _QWORD *v17; // rdi
  unsigned int v18; // eax
  int v19; // eax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  int v22; // ebx
  __int64 v23; // r12
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *i; // rdx
  _QWORD *v27; // rax

  v2 = *(_QWORD ***)(a1 + 216);
  v3 = *(_QWORD ***)(a1 + 224);
  if ( v2 != v3 )
  {
    v4 = *(_QWORD ***)(a1 + 216);
    do
    {
      v5 = (__int64)*v4;
      if ( *v4 )
      {
        sub_1D8E2D0(*v4);
        j_j___libc_free_0(v5, 72);
      }
      ++v4;
    }
    while ( v3 != v4 );
    *(_QWORD *)(a1 + 224) = v2;
  }
  v6 = *(_DWORD *)(a1 + 256);
  ++*(_QWORD *)(a1 + 240);
  if ( v6 )
  {
    v16 = 4 * v6;
    v7 = *(unsigned int *)(a1 + 264);
    if ( (unsigned int)(4 * v6) < 0x40 )
      v16 = 64;
    if ( v16 >= (unsigned int)v7 )
      goto LABEL_10;
    v17 = *(_QWORD **)(a1 + 248);
    v18 = v6 - 1;
    if ( v18 )
    {
      _BitScanReverse(&v18, v18);
      v19 = 1 << (33 - (v18 ^ 0x1F));
      if ( v19 < 64 )
        v19 = 64;
      if ( (_DWORD)v7 == v19 )
      {
        *(_QWORD *)(a1 + 256) = 0;
        v27 = &v17[2 * (unsigned int)v7];
        do
        {
          if ( v17 )
            *v17 = -8;
          v17 += 2;
        }
        while ( v27 != v17 );
        goto LABEL_13;
      }
      v20 = (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
          | (4 * v19 / 3u + 1)
          | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)
          | (((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
            | (4 * v19 / 3u + 1)
            | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4);
      v21 = (v20 >> 8) | v20;
      v22 = (v21 | (v21 >> 16)) + 1;
      v23 = 16 * ((v21 | (v21 >> 16)) + 1);
    }
    else
    {
      v23 = 2048;
      v22 = 128;
    }
    j___libc_free_0(v17);
    *(_DWORD *)(a1 + 264) = v22;
    v24 = (_QWORD *)sub_22077B0(v23);
    v25 = *(unsigned int *)(a1 + 264);
    *(_QWORD *)(a1 + 256) = 0;
    *(_QWORD *)(a1 + 248) = v24;
    for ( i = &v24[2 * v25]; i != v24; v24 += 2 )
    {
      if ( v24 )
        *v24 = -8;
    }
  }
  else if ( *(_DWORD *)(a1 + 260) )
  {
    v7 = *(unsigned int *)(a1 + 264);
    if ( (unsigned int)v7 <= 0x40 )
    {
LABEL_10:
      v8 = *(_QWORD **)(a1 + 248);
      for ( j = &v8[2 * v7]; j != v8; v8 += 2 )
        *v8 = -8;
      *(_QWORD *)(a1 + 256) = 0;
      goto LABEL_13;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 248));
    *(_QWORD *)(a1 + 248) = 0;
    *(_QWORD *)(a1 + 256) = 0;
    *(_DWORD *)(a1 + 264) = 0;
  }
LABEL_13:
  v10 = *(_QWORD *)(a1 + 160);
  result = *(unsigned int *)(a1 + 168);
  v12 = v10 + 8 * result;
LABEL_14:
  while ( v10 != v12 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD **)(v12 - 8);
      v12 -= 8;
      if ( !v13 )
        break;
      v14 = *(__int64 (__fastcall **)(_QWORD *))(*v13 + 8LL);
      if ( v14 != sub_1D59FF0 )
      {
        result = v14(v13);
        goto LABEL_14;
      }
      v15 = (_QWORD *)v13[1];
      *v13 = &unk_49F9CF0;
      if ( v15 != v13 + 3 )
        j_j___libc_free_0(v15, v13[3] + 1LL);
      result = j_j___libc_free_0(v13, 56);
      if ( v10 == v12 )
        goto LABEL_20;
    }
  }
LABEL_20:
  *(_DWORD *)(a1 + 168) = 0;
  return result;
}
