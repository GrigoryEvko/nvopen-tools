// Function: sub_FEF990
// Address: 0xfef990
//
__int64 __fastcall sub_FEF990(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // r15d
  __int64 result; // rax
  _QWORD *v8; // rbx
  unsigned int v9; // eax
  __int64 v10; // r14
  _QWORD *v11; // r13
  __int64 v12; // rdx
  unsigned int v13; // ecx
  unsigned int v14; // eax
  __int64 v15; // rdi
  int v16; // ebx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 j; // rdx
  void *v22; // r8
  __int64 v23; // rax
  unsigned int v24; // eax
  _QWORD *v25; // r13
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  char v30; // dl
  _QWORD *v31; // rbx
  char v32; // al
  __int64 v33; // rax
  bool v34; // zf
  __int64 v35; // rax
  void *v36; // [rsp+8h] [rbp-98h]
  _QWORD v37[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v38; // [rsp+28h] [rbp-78h]
  __int64 v39; // [rsp+30h] [rbp-70h]
  void *v40; // [rsp+40h] [rbp-60h]
  __int64 v41; // [rsp+48h] [rbp-58h] BYREF
  __int64 v42; // [rsp+50h] [rbp-50h]
  __int64 v43; // [rsp+58h] [rbp-48h]
  __int64 k; // [rsp+60h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 52) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 40), 24 * v3, 8);
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 48) = 0;
      *(_DWORD *)(a1 + 56) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v13 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v13 = 64;
  if ( (unsigned int)v3 <= v13 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 40);
    for ( i = &v4[3 * v3]; i != v4; *((_DWORD *)v4 - 4) = -1 )
    {
      *v4 = -4096;
      v4 += 3;
    }
    *(_QWORD *)(a1 + 48) = 0;
    goto LABEL_7;
  }
  v14 = v2 - 1;
  if ( !v14 )
  {
    v15 = *(_QWORD *)(a1 + 40);
    v16 = 64;
LABEL_33:
    sub_C7D6A0(v15, 24 * v3, 8);
    v17 = ((((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
             | (4 * v16 / 3u + 1)
             | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 16;
    v18 = (v17
         | (((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
             | (4 * v16 / 3u + 1)
             | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 56) = v18;
    v19 = sub_C7D670(24 * v18, 8);
    v20 = *(unsigned int *)(a1 + 56);
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 40) = v19;
    for ( j = v19 + 24 * v20; j != v19; v19 += 24 )
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = -4096;
        *(_DWORD *)(v19 + 8) = -1;
      }
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v14, v14);
  v15 = *(_QWORD *)(a1 + 40);
  v16 = 1 << (33 - (v14 ^ 0x1F));
  if ( v16 < 64 )
    v16 = 64;
  if ( (_DWORD)v3 != v16 )
    goto LABEL_33;
  *(_QWORD *)(a1 + 48) = 0;
  v35 = v15 + 24 * v3;
  do
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = -4096;
      *(_DWORD *)(v15 + 8) = -1;
    }
    v15 += 24;
  }
  while ( v35 != v15 );
LABEL_7:
  v6 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v6 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    v8 = *(_QWORD **)(a1 + 8);
    v9 = 4 * v6;
    v10 = 5LL * *(unsigned int *)(a1 + 24);
    if ( (unsigned int)(4 * v6) < 0x40 )
      v9 = 64;
    v11 = &v8[v10];
    if ( *(_DWORD *)(a1 + 24) > v9 )
    {
      v37[1] = 0;
      v37[0] = 2;
      v38 = -4096;
      v40 = &unk_49DE380;
      v39 = 0;
      v41 = 2;
      v22 = &unk_49DB368;
      v42 = 0;
      v43 = -8192;
      k = 0;
      do
      {
        v23 = v8[3];
        *v8 = v22;
        if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
        {
          v36 = v22;
          sub_BD60C0(v8 + 1);
          v22 = v36;
        }
        v8 += 5;
      }
      while ( v8 != v11 );
      v40 = &unk_49DB368;
      if ( v43 != 0 && v43 != -4096 && v43 != -8192 )
        sub_BD60C0(&v41);
      if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
        sub_BD60C0(v37);
      if ( v6 )
      {
        v24 = v6 - 1;
        v6 = 64;
        if ( v24 )
        {
          _BitScanReverse(&v24, v24);
          v6 = 1 << (33 - (v24 ^ 0x1F));
          if ( v6 < 64 )
            v6 = 64;
        }
      }
      v25 = *(_QWORD **)(a1 + 8);
      if ( *(_DWORD *)(a1 + 24) == v6 )
      {
        *(_QWORD *)(a1 + 16) = 0;
        v41 = 2;
        result = 5LL * (unsigned int)v6;
        v42 = 0;
        v31 = &v25[5 * (unsigned int)v6];
        v43 = -4096;
        v40 = &unk_49DE380;
        k = 0;
        if ( v31 != v25 )
        {
          do
          {
            if ( v25 )
            {
              v32 = v41;
              v25[2] = 0;
              v25[1] = v32 & 6;
              v33 = v43;
              v34 = v43 == -4096;
              v25[3] = v43;
              if ( v33 != 0 && !v34 && v33 != -8192 )
                sub_BD6050(v25 + 1, v41 & 0xFFFFFFFFFFFFFFF8LL);
              *v25 = &unk_49DE380;
              v25[4] = k;
            }
            v25 += 5;
          }
          while ( v31 != v25 );
          result = v43;
          v40 = &unk_49DB368;
          if ( v43 != 0 && v43 != -8192 && v43 != -4096 )
            return sub_BD60C0(&v41);
        }
      }
      else
      {
        result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v10 * 8, 8);
        if ( v6 )
        {
          v26 = ((((((((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
                   | (4 * v6 / 3u + 1)
                   | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
                 | (4 * v6 / 3u + 1)
                 | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
                 | (4 * v6 / 3u + 1)
                 | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 4)
               | (((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
               | (4 * v6 / 3u + 1)
               | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 16;
          v27 = (v26
               | (((((((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
                   | (4 * v6 / 3u + 1)
                   | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
                 | (4 * v6 / 3u + 1)
                 | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
                 | (4 * v6 / 3u + 1)
                 | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 4)
               | (((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
               | (4 * v6 / 3u + 1)
               | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(a1 + 24) = v27;
          result = sub_C7D670(40 * v27, 8);
          v28 = *(unsigned int *)(a1 + 24);
          *(_QWORD *)(a1 + 16) = 0;
          *(_QWORD *)(a1 + 8) = result;
          v41 = 2;
          v29 = result + 40 * v28;
          for ( k = 0; v29 != result; result += 40 )
          {
            if ( result )
            {
              v30 = v41;
              *(_QWORD *)(result + 16) = 0;
              *(_QWORD *)(result + 24) = -4096;
              *(_QWORD *)result = &unk_49DE380;
              *(_QWORD *)(result + 8) = v30 & 6;
              *(_QWORD *)(result + 32) = k;
            }
          }
        }
        else
        {
          *(_QWORD *)(a1 + 8) = 0;
          *(_QWORD *)(a1 + 16) = 0;
          *(_DWORD *)(a1 + 24) = 0;
        }
      }
    }
    else
    {
      v41 = 2;
      v42 = 0;
      v43 = -4096;
      v40 = &unk_49DE380;
      result = -4096;
      k = 0;
      if ( v8 == v11 )
      {
        *(_QWORD *)(a1 + 16) = 0;
        return result;
      }
      do
      {
        v12 = v8[3];
        if ( v12 != result )
        {
          if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
          {
            sub_BD60C0(v8 + 1);
            result = v43;
          }
          v8[3] = result;
          if ( result != 0 && result != -4096 && result != -8192 )
            sub_BD6050(v8 + 1, v41 & 0xFFFFFFFFFFFFFFF8LL);
          result = v43;
        }
        v8 += 5;
        *(v8 - 1) = k;
      }
      while ( v8 != v11 );
      *(_QWORD *)(a1 + 16) = 0;
      v40 = &unk_49DB368;
      if ( result != 0 && result != -4096 && result != -8192 )
        return sub_BD60C0(&v41);
    }
  }
  return result;
}
