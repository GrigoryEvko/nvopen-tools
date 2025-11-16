// Function: sub_18565C0
// Address: 0x18565c0
//
__int64 __fastcall sub_18565C0(_QWORD *a1)
{
  __int64 v1; // r13
  __int64 result; // rax
  int v3; // eax
  unsigned int v4; // ecx
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *i; // rdx
  int v8; // r15d
  _QWORD **v9; // rbx
  __int64 v10; // rdx
  _QWORD **v11; // r14
  _QWORD **v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r15
  __int64 v16; // rdi
  _QWORD **k; // rbx
  __int64 v18; // rax
  _QWORD *v19; // r12
  _QWORD *v20; // r8
  __int64 v21; // rdi
  int v22; // edx
  int v23; // ebx
  unsigned int v24; // r15d
  unsigned int v25; // eax
  _QWORD *v26; // rdi
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 m; // rdx
  _QWORD *v31; // rdi
  unsigned int v32; // eax
  int v33; // eax
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  int v36; // ebx
  __int64 v37; // r12
  _QWORD *v38; // rax
  __int64 v39; // rdx
  _QWORD *j; // rdx
  _QWORD *v41; // rax
  _QWORD *v42; // [rsp+8h] [rbp-38h]

  v1 = a1[1];
  result = (__int64)&unk_49F13A0;
  *a1 = &unk_49F13A0;
  if ( !v1 )
    return result;
  v3 = *(_DWORD *)(v1 + 80);
  ++*(_QWORD *)(v1 + 64);
  if ( !v3 )
  {
    if ( !*(_DWORD *)(v1 + 84) )
      goto LABEL_9;
    v5 = *(unsigned int *)(v1 + 88);
    if ( (unsigned int)v5 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(v1 + 72));
      *(_QWORD *)(v1 + 72) = 0;
      *(_QWORD *)(v1 + 80) = 0;
      *(_DWORD *)(v1 + 88) = 0;
      goto LABEL_9;
    }
    goto LABEL_6;
  }
  v4 = 4 * v3;
  v5 = *(unsigned int *)(v1 + 88);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v4 = 64;
  if ( (unsigned int)v5 <= v4 )
  {
LABEL_6:
    v6 = *(_QWORD **)(v1 + 72);
    for ( i = &v6[3 * v5]; i != v6; *(v6 - 2) = -8 )
    {
      *v6 = -8;
      v6 += 3;
    }
    *(_QWORD *)(v1 + 80) = 0;
    goto LABEL_9;
  }
  v31 = *(_QWORD **)(v1 + 72);
  v32 = v3 - 1;
  if ( !v32 )
  {
    v37 = 3072;
    v36 = 128;
LABEL_55:
    j___libc_free_0(v31);
    *(_DWORD *)(v1 + 88) = v36;
    v38 = (_QWORD *)sub_22077B0(v37);
    v39 = *(unsigned int *)(v1 + 88);
    *(_QWORD *)(v1 + 80) = 0;
    *(_QWORD *)(v1 + 72) = v38;
    for ( j = &v38[3 * v39]; j != v38; v38 += 3 )
    {
      if ( v38 )
      {
        *v38 = -8;
        v38[1] = -8;
      }
    }
    goto LABEL_9;
  }
  _BitScanReverse(&v32, v32);
  v33 = 1 << (33 - (v32 ^ 0x1F));
  if ( v33 < 64 )
    v33 = 64;
  if ( (_DWORD)v5 != v33 )
  {
    v34 = ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)
        | (4 * v33 / 3u + 1)
        | ((((unsigned __int64)(4 * v33 / 3u + 1) >> 1) | (4 * v33 / 3u + 1)) >> 2);
    v35 = (((v34 | (v34 >> 4)) >> 8) | v34 | (v34 >> 4) | ((((v34 | (v34 >> 4)) >> 8) | v34 | (v34 >> 4)) >> 16)) + 1;
    v36 = v35;
    v37 = 24 * v35;
    goto LABEL_55;
  }
  *(_QWORD *)(v1 + 80) = 0;
  v41 = &v31[3 * v5];
  do
  {
    if ( v31 )
    {
      *v31 = -8;
      v31[1] = -8;
    }
    v31 += 3;
  }
  while ( v41 != v31 );
LABEL_9:
  v8 = *(_DWORD *)(v1 + 48);
  ++*(_QWORD *)(v1 + 32);
  if ( v8 || (result = *(unsigned int *)(v1 + 52), (_DWORD)result) )
  {
    v9 = *(_QWORD ***)(v1 + 40);
    result = (unsigned int)(4 * v8);
    v10 = *(unsigned int *)(v1 + 56);
    v11 = &v9[4 * v10];
    if ( (unsigned int)result < 0x40 )
      result = 64;
    if ( (unsigned int)v10 <= (unsigned int)result )
    {
      v12 = v9 + 1;
      if ( v9 != v11 )
      {
        while ( 1 )
        {
          v13 = (__int64)*(v12 - 1);
          if ( v13 != -8 )
          {
            if ( v13 != -16 )
            {
              v14 = *v12;
              while ( v14 != v12 )
              {
                v15 = v14;
                v14 = (_QWORD *)*v14;
                v16 = v15[3];
                if ( v16 )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
                j_j___libc_free_0(v15, 32);
              }
            }
            *(v12 - 1) = (_QWORD *)-8LL;
          }
          result = (__int64)(v12 + 4);
          if ( v11 == v12 + 3 )
            break;
          v12 += 4;
        }
      }
LABEL_24:
      *(_QWORD *)(v1 + 48) = 0;
      return result;
    }
    for ( k = v9 + 1; ; k += 4 )
    {
      v18 = (__int64)*(k - 1);
      if ( v18 != -8 && v18 != -16 )
      {
        v19 = *k;
        while ( k != v19 )
        {
          v20 = v19;
          v19 = (_QWORD *)*v19;
          v21 = v20[3];
          if ( v21 )
          {
            v42 = v20;
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
            v20 = v42;
          }
          j_j___libc_free_0(v20, 32);
        }
      }
      result = (__int64)(k + 4);
      if ( v11 == k + 3 )
        break;
    }
    v22 = *(_DWORD *)(v1 + 56);
    if ( !v8 )
    {
      if ( v22 )
      {
        result = j___libc_free_0(*(_QWORD *)(v1 + 40));
        *(_QWORD *)(v1 + 40) = 0;
        *(_QWORD *)(v1 + 48) = 0;
        *(_DWORD *)(v1 + 56) = 0;
        return result;
      }
      goto LABEL_24;
    }
    v23 = 64;
    v24 = v8 - 1;
    if ( v24 )
    {
      _BitScanReverse(&v25, v24);
      v23 = 1 << (33 - (v25 ^ 0x1F));
      if ( v23 < 64 )
        v23 = 64;
    }
    v26 = *(_QWORD **)(v1 + 40);
    if ( v23 == v22 )
    {
      *(_QWORD *)(v1 + 48) = 0;
      result = (__int64)&v26[4 * (unsigned int)v23];
      do
      {
        if ( v26 )
          *v26 = -8;
        v26 += 4;
      }
      while ( (_QWORD *)result != v26 );
    }
    else
    {
      j___libc_free_0(v26);
      v27 = ((((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
               | (4 * v23 / 3u + 1)
               | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
             | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 16;
      v28 = (v27
           | (((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
               | (4 * v23 / 3u + 1)
               | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
             | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(v1 + 56) = v28;
      result = sub_22077B0(32 * v28);
      v29 = *(unsigned int *)(v1 + 56);
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 40) = result;
      for ( m = result + 32 * v29; m != result; result += 32 )
      {
        if ( result )
          *(_QWORD *)result = -8;
      }
    }
  }
  return result;
}
