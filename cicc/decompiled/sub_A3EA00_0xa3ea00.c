// Function: sub_A3EA00
// Address: 0xa3ea00
//
__int64 __fastcall sub_A3EA00(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned int *a8,
        _BYTE *a9)
{
  __int64 *v9; // r14
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 *i; // rbx
  int v15; // edx
  __int64 v16; // rdi
  __int64 *v17; // rsi
  __int64 v18; // r10
  int v19; // edx
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r14
  unsigned int v23; // r11d
  __int64 v24; // rdi
  unsigned int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // r14
  unsigned int v28; // eax
  unsigned int v30; // r14d
  unsigned int v31; // eax
  int v32; // eax
  unsigned int v33; // edx
  int v34; // eax
  unsigned int v35; // r14d
  unsigned int v36; // eax
  int v37; // r11d
  unsigned int v38; // [rsp+4h] [rbp-4Ch]
  _BYTE *v39; // [rsp+10h] [rbp-40h]
  _BYTE *v40; // [rsp+10h] [rbp-40h]
  unsigned int *v41; // [rsp+18h] [rbp-38h]
  unsigned int *v42; // [rsp+18h] [rbp-38h]
  int v43; // [rsp+18h] [rbp-38h]

  v9 = a1;
  v10 = *a1;
  v11 = *(a1 - 2);
  v38 = *((_DWORD *)a1 + 2);
  if ( v11 != *a1 )
  {
    for ( i = a1 - 2; ; i -= 2 )
    {
      v15 = *(_DWORD *)(a7 + 24);
      v16 = *(_QWORD *)(v10 + 24);
      v17 = i + 2;
      v18 = *(_QWORD *)(a7 + 8);
      if ( v15 )
      {
        v19 = v15 - 1;
        v20 = v19 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v21 = (__int64 *)(v18 + 16LL * v20);
        v22 = *v21;
        if ( v16 == *v21 )
        {
LABEL_5:
          v23 = *((_DWORD *)v21 + 2);
          v24 = *(_QWORD *)(v11 + 24);
        }
        else
        {
          v34 = 1;
          while ( v22 != -4096 )
          {
            v37 = v34 + 1;
            v20 = v19 & (v34 + v20);
            v21 = (__int64 *)(v18 + 16LL * v20);
            v22 = *v21;
            if ( v16 == *v21 )
              goto LABEL_5;
            v34 = v37;
          }
          v24 = *(_QWORD *)(v11 + 24);
          v23 = 0;
        }
        v25 = v19 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v26 = (__int64 *)(v18 + 16LL * v25);
        v27 = *v26;
        if ( *v26 == v24 )
        {
LABEL_7:
          v28 = *((_DWORD *)v26 + 2);
          if ( v28 > v23 )
          {
            if ( v28 > *a8 || *a9 )
              goto LABEL_10;
            goto LABEL_15;
          }
        }
        else
        {
          v32 = 1;
          while ( v27 != -4096 )
          {
            v25 = v19 & (v32 + v25);
            v43 = v32 + 1;
            v26 = (__int64 *)(v18 + 16LL * v25);
            v27 = *v26;
            if ( *v26 == v24 )
              goto LABEL_7;
            v32 = v43;
          }
          v28 = 0;
        }
        v33 = *a8;
        if ( v28 < v23 )
        {
          if ( v33 >= v23 && !*a9 )
          {
LABEL_10:
            v9 = v17;
            break;
          }
          goto LABEL_15;
        }
        if ( v33 < v23 )
          goto LABEL_14;
      }
      if ( *a9 )
      {
LABEL_14:
        v39 = a9;
        v41 = a8;
        v30 = sub_BD2910(v10);
        v31 = sub_BD2910(v11);
        a8 = v41;
        a9 = v39;
        v17 = i + 2;
        if ( v30 <= v31 )
          goto LABEL_10;
        goto LABEL_15;
      }
      v40 = a9;
      v42 = a8;
      v35 = sub_BD2910(v10);
      v36 = sub_BD2910(v11);
      a8 = v42;
      a9 = v40;
      v17 = i + 2;
      if ( v35 >= v36 )
        goto LABEL_10;
LABEL_15:
      v11 = *(i - 2);
      i[2] = *i;
      *((_DWORD *)i + 6) = *((_DWORD *)i + 2);
      if ( v11 == v10 )
      {
        v9 = i;
        break;
      }
    }
  }
  *v9 = v10;
  *((_DWORD *)v9 + 2) = v38;
  return v38;
}
