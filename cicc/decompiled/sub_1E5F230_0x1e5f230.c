// Function: sub_1E5F230
// Address: 0x1e5f230
//
void __fastcall sub_1E5F230(__int64 a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // rdi
  _QWORD *v6; // rbx
  _QWORD *v7; // rbx
  int v8; // r14d
  _QWORD *v9; // rbx
  __int64 v10; // rdx
  _QWORD *v11; // r13
  unsigned int v12; // eax
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  _BYTE *v15; // rcx
  size_t v16; // rdx
  _QWORD *v17; // rdi
  int v18; // edx
  int v19; // ebx
  unsigned int v20; // r14d
  unsigned int v21; // eax
  unsigned int v22; // eax
  size_t v23; // rbx
  _BYTE *v24; // r13
  __int64 src; // [rsp+8h] [rbp-28h] BYREF
  _BYTE v26[32]; // [rsp+10h] [rbp-20h] BYREF

  v2 = *(_QWORD **)a1;
  v3 = *(_QWORD *)(a1 + 16);
  src = 0;
  if ( (unsigned __int64)(v3 - (_QWORD)v2) > 7 )
  {
    v15 = *(_BYTE **)(a1 + 8);
    v16 = v15 - (_BYTE *)v2;
    if ( (unsigned __int64)(v15 - (_BYTE *)v2) <= 7 )
    {
      v23 = 8;
      v24 = &v26[v16 - 8];
      if ( &v26[v16 - 8] == (_BYTE *)&src
        || (memcpy(v2, &src, v16), v15 = *(_BYTE **)(a1 + 8), v23 = v26 - v24, v24 != v26) )
      {
        v15 = memcpy(v15, v24, v23);
      }
      *(_QWORD *)(a1 + 8) = &v15[v23];
    }
    else
    {
      *v2 = 0;
      v17 = v2 + 1;
      if ( v17 != *(_QWORD **)(a1 + 8) )
        *(_QWORD *)(a1 + 8) = v17;
    }
  }
  else
  {
    v4 = (_QWORD *)sub_22077B0(8);
    v5 = *(_QWORD **)a1;
    *v4 = 0;
    v6 = v4;
    if ( v5 )
      j_j___libc_free_0(v5, *(_QWORD *)(a1 + 16) - (_QWORD)v5);
    *(_QWORD *)a1 = v6;
    v7 = v6 + 1;
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = v7;
  }
  v8 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  if ( v8 || *(_DWORD *)(a1 + 44) )
  {
    v9 = *(_QWORD **)(a1 + 32);
    v10 = *(unsigned int *)(a1 + 48);
    v11 = &v9[9 * v10];
    v12 = 4 * v8;
    if ( (unsigned int)(4 * v8) < 0x40 )
      v12 = 64;
    if ( (unsigned int)v10 <= v12 )
    {
      for ( ; v9 != v11; v9 += 9 )
      {
        if ( *v9 != -8 )
        {
          if ( *v9 != -16 )
          {
            v13 = v9[5];
            if ( (_QWORD *)v13 != v9 + 7 )
              _libc_free(v13);
          }
          *v9 = -8;
        }
      }
      goto LABEL_17;
    }
    do
    {
      if ( *v9 != -16 && *v9 != -8 )
      {
        v14 = v9[5];
        if ( (_QWORD *)v14 != v9 + 7 )
          _libc_free(v14);
      }
      v9 += 9;
    }
    while ( v9 != v11 );
    v18 = *(_DWORD *)(a1 + 48);
    if ( v8 )
    {
      v19 = 64;
      v20 = v8 - 1;
      if ( v20 )
      {
        _BitScanReverse(&v21, v20);
        v19 = 1 << (33 - (v21 ^ 0x1F));
        if ( v19 < 64 )
          v19 = 64;
      }
      if ( v19 == v18 )
        goto LABEL_35;
      j___libc_free_0(*(_QWORD *)(a1 + 32));
      v22 = sub_1454B60(4 * v19 / 3u + 1);
      *(_DWORD *)(a1 + 48) = v22;
      if ( v22 )
      {
        *(_QWORD *)(a1 + 32) = sub_22077B0(72LL * v22);
LABEL_35:
        sub_1E5F1F0(a1 + 24);
        return;
      }
    }
    else
    {
      if ( !v18 )
        goto LABEL_35;
      j___libc_free_0(*(_QWORD *)(a1 + 32));
      *(_DWORD *)(a1 + 48) = 0;
    }
    *(_QWORD *)(a1 + 32) = 0;
LABEL_17:
    *(_QWORD *)(a1 + 40) = 0;
  }
}
