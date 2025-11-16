// Function: sub_1FD5930
// Address: 0x1fd5930
//
__int64 __fastcall sub_1FD5930(__int64 a1)
{
  __int64 v2; // r15
  __int64 v3; // r12
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // r8d
  int v11; // eax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *i; // rdx
  __int64 v15; // rax
  __int64 result; // rax
  unsigned int v17; // ecx
  _QWORD *v18; // rdi
  unsigned int v19; // eax
  int v20; // eax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  int v23; // ebx
  __int64 v24; // r12
  _QWORD *v25; // rax
  __int64 v26; // rdx
  _QWORD *j; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // rax
  char v32; // [rsp+Fh] [rbp-61h] BYREF
  __int64 v33; // [rsp+10h] [rbp-60h] BYREF
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+20h] [rbp-50h]
  int v36; // [rsp+28h] [rbp-48h]
  __int64 v37; // [rsp+30h] [rbp-40h]
  int v38; // [rsp+38h] [rbp-38h]

  if ( byte_4FCEB00 )
  {
    v2 = *(_QWORD *)(a1 + 144);
    v3 = *(_QWORD *)(a1 + 152);
    if ( v2 != v3 )
    {
      if ( v3 )
      {
        v33 = 0;
        v34 = 0;
        v35 = 0;
        v36 = 0;
        v37 = 0;
        v38 = -1;
      }
      else
      {
        v28 = *(_QWORD *)(a1 + 40);
        v33 = 0;
        v34 = 0;
        v29 = *(_QWORD *)(v28 + 784);
        v35 = 0;
        v36 = 0;
        v30 = v29 + 24;
        v37 = 0;
        v38 = -1;
        if ( v2 == v30 )
        {
LABEL_23:
          j___libc_free_0(v3);
          goto LABEL_24;
        }
        v3 = v30;
      }
      while ( 1 )
      {
        v4 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v4 )
          BUG();
        v5 = *(_QWORD *)v4;
        v6 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v4 & 4) == 0 && (*(_BYTE *)(v4 + 46) & 4) != 0 )
        {
          while ( 1 )
          {
            v7 = v5 & 0xFFFFFFFFFFFFFFF8LL;
            v6 = v7;
            if ( (*(_BYTE *)(v7 + 46) & 4) == 0 )
              break;
            v5 = *(_QWORD *)v7;
          }
        }
        v32 = 1;
        if ( (unsigned __int8)sub_1E17B50(v2, 0, &v32) )
        {
          v8 = *(_QWORD *)(v2 + 32);
          v9 = v8 + 40LL * *(unsigned int *)(v2 + 40);
          if ( v8 != v9 )
          {
            v10 = 0;
            do
            {
              if ( !*(_BYTE *)v8 )
              {
                if ( (*(_BYTE *)(v8 + 3) & 0x10) != 0 )
                {
                  if ( v10 )
                    goto LABEL_20;
                  v10 = *(_DWORD *)(v8 + 8);
                }
                else if ( *(int *)(v8 + 8) < 0 )
                {
                  goto LABEL_20;
                }
              }
              v8 += 40;
            }
            while ( v9 != v8 );
            if ( v10 )
              sub_1FD5250((_QWORD *)a1, (unsigned __int8 *)v2, v10, (__int64)&v33);
          }
        }
LABEL_20:
        if ( v3 == v6 )
          break;
        v2 = v6;
      }
      v3 = v34;
      goto LABEL_23;
    }
  }
LABEL_24:
  v11 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v11 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_30;
    v12 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v12 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 16));
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_30;
    }
    goto LABEL_27;
  }
  v17 = 4 * v11;
  v12 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v11) < 0x40 )
    v17 = 64;
  if ( (unsigned int)v12 <= v17 )
  {
LABEL_27:
    v13 = *(_QWORD **)(a1 + 16);
    for ( i = &v13[2 * v12]; i != v13; v13 += 2 )
      *v13 = -8;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_30;
  }
  v18 = *(_QWORD **)(a1 + 16);
  v19 = v11 - 1;
  if ( !v19 )
  {
    v24 = 2048;
    v23 = 128;
LABEL_39:
    j___libc_free_0(v18);
    *(_DWORD *)(a1 + 32) = v23;
    v25 = (_QWORD *)sub_22077B0(v24);
    v26 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v25;
    for ( j = &v25[2 * v26]; j != v25; v25 += 2 )
    {
      if ( v25 )
        *v25 = -8;
    }
    goto LABEL_30;
  }
  _BitScanReverse(&v19, v19);
  v20 = 1 << (33 - (v19 ^ 0x1F));
  if ( v20 < 64 )
    v20 = 64;
  if ( (_DWORD)v12 != v20 )
  {
    v21 = (4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1);
    v22 = ((v21 | (v21 >> 2)) >> 4) | v21 | (v21 >> 2) | ((((v21 | (v21 >> 2)) >> 4) | v21 | (v21 >> 2)) >> 8);
    v23 = (v22 | (v22 >> 16)) + 1;
    v24 = 16 * ((v22 | (v22 >> 16)) + 1);
    goto LABEL_39;
  }
  *(_QWORD *)(a1 + 24) = 0;
  v31 = &v18[2 * (unsigned int)v12];
  do
  {
    if ( v18 )
      *v18 = -8;
    v18 += 2;
  }
  while ( v31 != v18 );
LABEL_30:
  *(_QWORD *)(a1 + 144) = *(_QWORD *)(a1 + 152);
  sub_1FD3A30(a1);
  v15 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 168) = *(_QWORD *)(v15 + 792);
  result = *(_QWORD *)(v15 + 792);
  *(_QWORD *)(a1 + 160) = result;
  return result;
}
