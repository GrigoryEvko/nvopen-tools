// Function: sub_2680390
// Address: 0x2680390
//
__int64 __fastcall sub_2680390(__int64 *a1, int a2)
{
  __int64 v2; // r10
  __int64 v4; // r13
  __int64 v5; // r8
  int v6; // ecx
  _QWORD *v7; // rbx
  unsigned int v8; // eax
  __int64 v9; // r12
  _QWORD *v10; // r15
  volatile signed __int32 *v11; // r12
  signed __int32 v12; // eax
  __int64 result; // rax
  signed __int32 v14; // eax
  signed __int32 v15; // eax
  volatile signed __int32 *v16; // rdi
  int v17; // edx
  __int64 v18; // rbx
  unsigned int v19; // ecx
  unsigned int v20; // eax
  _QWORD *v21; // rdi
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *i; // rdx
  signed __int32 v27; // eax
  _QWORD *v28; // rax
  __int64 v29; // [rsp+8h] [rbp-58h]
  int v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  __int64 v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+20h] [rbp-40h]
  __int64 v37; // [rsp+20h] [rbp-40h]
  __int64 *v38; // [rsp+28h] [rbp-38h]

  v2 = a2;
  v4 = 20LL * a2;
  v5 = (__int64)&a1[v4 + 455];
  v6 = a1[v4 + 457];
  ++*(_QWORD *)v5;
  v38 = a1 + 439;
  if ( !v6 && !HIDWORD(a1[v4 + 457]) )
    goto LABEL_17;
  v7 = (_QWORD *)a1[v4 + 456];
  v8 = 4 * v6;
  v9 = 24LL * LODWORD(a1[v4 + 458]);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v8 = 64;
  v10 = &v7[(unsigned __int64)v9 / 8];
  if ( LODWORD(a1[v4 + 458]) <= v8 )
  {
    while ( v7 != v10 )
    {
      if ( *v7 != -4096 )
      {
        if ( *v7 != -8192 )
        {
          v11 = (volatile signed __int32 *)v7[2];
          if ( v11 )
          {
            if ( &_pthread_key_create )
            {
              v12 = _InterlockedExchangeAdd(v11 + 2, 0xFFFFFFFF);
            }
            else
            {
              v12 = *((_DWORD *)v11 + 2);
              *((_DWORD *)v11 + 2) = v12 - 1;
            }
            if ( v12 == 1 )
            {
              v31 = v2;
              v35 = v5;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 16LL))(v11);
              v5 = v35;
              v2 = v31;
              if ( &_pthread_key_create )
              {
                v14 = _InterlockedExchangeAdd(v11 + 3, 0xFFFFFFFF);
              }
              else
              {
                v14 = *((_DWORD *)v11 + 3);
                *((_DWORD *)v11 + 3) = v14 - 1;
              }
              if ( v14 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 24LL))(v11);
                v2 = v31;
                v5 = v35;
              }
            }
          }
        }
        *v7 = -4096;
      }
      v7 += 3;
    }
    goto LABEL_16;
  }
  do
  {
    while ( *v7 == -4096 || *v7 == -8192 )
    {
LABEL_27:
      v7 += 3;
      if ( v7 == v10 )
        goto LABEL_35;
    }
    v16 = (volatile signed __int32 *)v7[2];
    if ( v16 )
    {
      if ( &_pthread_key_create )
      {
        v15 = _InterlockedExchangeAdd(v16 + 2, 0xFFFFFFFF);
      }
      else
      {
        v15 = *((_DWORD *)v16 + 2);
        *((_DWORD *)v16 + 2) = v15 - 1;
      }
      if ( v15 == 1 )
      {
        v29 = v2;
        v30 = v6;
        v34 = v5;
        (*(void (**)(void))(*(_QWORD *)v16 + 16LL))();
        v5 = v34;
        v6 = v30;
        v2 = v29;
        if ( &_pthread_key_create )
        {
          v27 = _InterlockedExchangeAdd(v16 + 3, 0xFFFFFFFF);
        }
        else
        {
          v27 = *((_DWORD *)v16 + 3);
          *((_DWORD *)v16 + 3) = v27 - 1;
        }
        if ( v27 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 24LL))(v16);
          v2 = v29;
          v6 = v30;
          v5 = v34;
        }
      }
      goto LABEL_27;
    }
    v7 += 3;
  }
  while ( v7 != v10 );
LABEL_35:
  v17 = *(_DWORD *)(v5 + 24);
  if ( !v6 )
  {
    if ( v17 )
    {
      v33 = v2;
      v37 = v5;
      sub_C7D6A0(*(_QWORD *)(v5 + 8), v9, 8);
      v2 = v33;
      *(_QWORD *)(v37 + 8) = 0;
      *(_QWORD *)(v37 + 16) = 0;
      *(_DWORD *)(v37 + 24) = 0;
      goto LABEL_17;
    }
LABEL_16:
    *(_QWORD *)(v5 + 16) = 0;
    goto LABEL_17;
  }
  v18 = 64;
  v19 = v6 - 1;
  if ( v19 )
  {
    _BitScanReverse(&v20, v19);
    v18 = (unsigned int)(1 << (33 - (v20 ^ 0x1F)));
    if ( (int)v18 < 64 )
      v18 = 64;
  }
  v21 = *(_QWORD **)(v5 + 8);
  if ( (_DWORD)v18 == v17 )
  {
    *(_QWORD *)(v5 + 16) = 0;
    v28 = &v21[3 * v18];
    do
    {
      if ( v21 )
        *v21 = -4096;
      v21 += 3;
    }
    while ( v28 != v21 );
  }
  else
  {
    v32 = v2;
    v36 = v5;
    sub_C7D6A0((__int64)v21, v9, 8);
    v22 = ((((((((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v18 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v18 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v18 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v18 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 16;
    v23 = (v22
         | (((((((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v18 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v18 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v18 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v18 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(v36 + 24) = v23;
    v24 = (_QWORD *)sub_C7D670(24 * v23, 8);
    v2 = v32;
    v25 = *(unsigned int *)(v36 + 24);
    *(_QWORD *)(v36 + 8) = v24;
    *(_QWORD *)(v36 + 16) = 0;
    for ( i = &v24[3 * v25]; i != v24; v24 += 3 )
    {
      if ( v24 )
        *v24 = -4096;
    }
  }
LABEL_17:
  result = (__int64)&a1[20 * v2];
  if ( *(_QWORD *)(result + 3632) )
  {
    sub_3122A50(a1 + 50, *(unsigned int *)(result + 3512));
    return sub_267FDF0(a1, (__int64)&v38[v4]);
  }
  return result;
}
