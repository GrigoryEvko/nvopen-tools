// Function: sub_1DB7C30
// Address: 0x1db7c30
//
_QWORD *__fastcall sub_1DB7C30(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v9; // rbx
  _QWORD *result; // rax
  unsigned __int64 v11; // r9
  __int64 v12; // rsi
  _QWORD *v13; // rax
  unsigned int v14; // ecx
  _QWORD *v15; // rdi
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // r10
  __int64 v19; // rbx
  unsigned int v20; // eax
  unsigned __int64 v21; // r11
  __int64 v22; // r10
  unsigned __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 *v26; // rsi
  unsigned int v27; // eax
  __int64 *v28; // r8
  unsigned __int64 v29; // [rsp+8h] [rbp-58h]
  unsigned __int64 v30; // [rsp+8h] [rbp-58h]
  unsigned int v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  __int64 *v35[7]; // [rsp+28h] [rbp-38h] BYREF

  v9 = *(_QWORD **)(a1 + 96);
  v35[0] = (__int64 *)a1;
  if ( !v9 )
  {
    if ( *(_DWORD *)(a1 + 8) )
    {
      v21 = a5 & 0xFFFFFFFFFFFFFFF8LL;
      v33 = (a5 >> 1) & 3;
      if ( ((a5 >> 1) & 3) != 0 )
      {
        v22 = v21 | (2LL * ((int)v33 - 1));
        v23 = a5 & 0xFFFFFFFFFFFFFFF8LL | (2LL * ((int)v33 - 1)) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v23 = *(_QWORD *)v21 & 0xFFFFFFFFFFFFFFF8LL;
        v22 = v23 | 6;
      }
      v24 = *(_QWORD *)a1;
      v25 = *(unsigned int *)(a1 + 8);
      do
      {
        v26 = (__int64 *)(v24 + 8 * ((v25 >> 1) + (v25 & 0xFFFFFFFFFFFFFFFELL)));
        if ( (*(_DWORD *)(v23 + 24) | (unsigned int)(v22 >> 1) & 3) >= (*(_DWORD *)((*v26 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                      | (unsigned int)(*v26 >> 1) & 3) )
        {
          v24 = (__int64)(v26 + 3);
          v25 = v25 - (v25 >> 1) - 1;
        }
        else
        {
          v25 >>= 1;
        }
      }
      while ( v25 > 0 );
      if ( *(_QWORD *)a1 == v24
        || (v27 = *(_DWORD *)((*(_QWORD *)(v24 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v24 - 16) >> 1) & 3,
            v27 <= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3)) )
      {
        sub_1DB3940(a2, (__int64)&a2[a3], a4, v22);
      }
      else
      {
        if ( v27 < (*(_DWORD *)(v21 + 24) | (unsigned int)v33) )
        {
          v34 = v24;
          if ( &a2[a3] != sub_1DB3940(a2, (__int64)&a2[a3], *(_QWORD *)(v24 - 16), v22) )
            return v9;
          sub_1DB37E0(v35, (_QWORD *)(v34 - 24), a5);
          v24 = v34;
        }
        return *(_QWORD **)(v24 - 8);
      }
    }
    return v9;
  }
  result = 0;
  if ( !v9[5] )
    return result;
  v11 = a5 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = (a5 >> 1) & 3;
  v31 = (a5 >> 1) & 3;
  if ( v12 )
    v32 = v11 | (2LL * ((int)v12 - 1));
  else
    v32 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL | 6;
  v13 = (_QWORD *)v9[2];
  if ( v13 )
  {
    v14 = *(_DWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v32 >> 1) & 3;
    v15 = v9 + 1;
    do
    {
      while ( 1 )
      {
        v16 = *(_DWORD *)((v13[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v13[4] >> 1) & 3;
        if ( v16 > v14
          || v16 >= v14
          && ((unsigned int)v12 | *(_DWORD *)(v11 + 24)) < (*(_DWORD *)((v13[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                          | (unsigned int)((__int64)v13[5] >> 1) & 3) )
        {
          break;
        }
        v13 = (_QWORD *)v13[3];
        if ( !v13 )
          goto LABEL_13;
      }
      v15 = v13;
      v13 = (_QWORD *)v13[2];
    }
    while ( v13 );
LABEL_13:
    if ( v9 + 1 != v15
      && v14 >= (*(_DWORD *)((v15[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v15[4] >> 1) & 3) )
    {
      v29 = a5 & 0xFFFFFFFFFFFFFFF8LL;
      v17 = sub_220EF30(v15);
      v11 = v29;
      v15 = (_QWORD *)v17;
    }
  }
  else
  {
    v15 = v9 + 1;
  }
  if ( (_QWORD *)v9[3] == v15
    || (v30 = v11,
        v19 = sub_220EFE0(v15),
        v18 = *(_QWORD *)(v19 + 40),
        v20 = *(_DWORD *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v18 >> 1) & 3,
        v20 <= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3)) )
  {
    sub_1DB3940(a2, (__int64)&a2[a3], a4, v32);
    return 0;
  }
  if ( v20 >= (*(_DWORD *)(v30 + 24) | v31) )
    return *(_QWORD **)(v19 + 48);
  v28 = sub_1DB3940(a2, (__int64)&a2[a3], v18, v32);
  result = 0;
  if ( &a2[a3] == v28 )
  {
    sub_1DB7AE0((__int64)v35, v19, a5);
    return *(_QWORD **)(v19 + 48);
  }
  return result;
}
