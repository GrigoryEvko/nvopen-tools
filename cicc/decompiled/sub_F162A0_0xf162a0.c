// Function: sub_F162A0
// Address: 0xf162a0
//
unsigned __int8 *__fastcall sub_F162A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  int v6; // r11d
  __int64 v7; // r9
  _QWORD *v8; // rdx
  unsigned int v9; // edi
  _QWORD *v10; // rax
  __int64 v11; // rcx
  unsigned int v12; // esi
  __int64 v13; // r13
  __int64 v14; // r8
  int v15; // esi
  int v16; // esi
  unsigned int v17; // ecx
  int v18; // eax
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned __int8 *v21; // r11
  __int64 v22; // r10
  int v24; // eax
  int v25; // ecx
  int v26; // ecx
  unsigned int v27; // r14d
  __int64 v28; // rdi
  int v29; // r10d
  __int64 v30; // rsi
  __int64 v31; // rax
  int v32; // r14d
  _QWORD *v33; // r10
  unsigned int v34; // [rsp+0h] [rbp-50h]
  unsigned int v35; // [rsp+0h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-40h]
  unsigned __int8 *v37; // [rsp+10h] [rbp-40h]
  unsigned __int8 *v39; // [rsp+18h] [rbp-38h]
  __int64 v40; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 16);
  if ( !v3 )
    return (unsigned __int8 *)v3;
  v4 = *(_QWORD *)(a1 + 40);
  v36 = v4 + 2064;
  do
  {
    while ( 1 )
    {
      v12 = *(_DWORD *)(v4 + 2088);
      v13 = *(_QWORD *)(v3 + 24);
      v14 = *(unsigned int *)(v4 + 8);
      if ( !v12 )
      {
        ++*(_QWORD *)(v4 + 2064);
LABEL_7:
        v34 = v14;
        sub_9BAAD0(v36, 2 * v12);
        v15 = *(_DWORD *)(v4 + 2088);
        if ( !v15 )
          goto LABEL_54;
        v16 = v15 - 1;
        v7 = *(_QWORD *)(v4 + 2072);
        v14 = v34;
        v17 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v18 = *(_DWORD *)(v4 + 2080) + 1;
        v8 = (_QWORD *)(v7 + 16LL * v17);
        v19 = *v8;
        if ( v13 != *v8 )
        {
          v32 = 1;
          v33 = 0;
          while ( v19 != -4096 )
          {
            if ( v19 == -8192 && !v33 )
              v33 = v8;
            v17 = v16 & (v32 + v17);
            v8 = (_QWORD *)(v7 + 16LL * v17);
            v19 = *v8;
            if ( v13 == *v8 )
              goto LABEL_9;
            ++v32;
          }
          if ( v33 )
            v8 = v33;
        }
        goto LABEL_9;
      }
      v6 = 1;
      v7 = *(_QWORD *)(v4 + 2072);
      v8 = 0;
      v9 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v10 = (_QWORD *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( v13 != *v10 )
        break;
LABEL_4:
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        goto LABEL_14;
    }
    while ( v11 != -4096 )
    {
      if ( v8 || v11 != -8192 )
        v10 = v8;
      v9 = (v12 - 1) & (v6 + v9);
      v11 = *(_QWORD *)(v7 + 16LL * v9);
      if ( v13 == v11 )
        goto LABEL_4;
      ++v6;
      v8 = v10;
      v10 = (_QWORD *)(v7 + 16LL * v9);
    }
    if ( !v8 )
      v8 = v10;
    v24 = *(_DWORD *)(v4 + 2080);
    ++*(_QWORD *)(v4 + 2064);
    v18 = v24 + 1;
    if ( 4 * v18 >= 3 * v12 )
      goto LABEL_7;
    if ( v12 - *(_DWORD *)(v4 + 2084) - v18 <= v12 >> 3 )
    {
      v35 = v14;
      sub_9BAAD0(v36, v12);
      v25 = *(_DWORD *)(v4 + 2088);
      if ( !v25 )
      {
LABEL_54:
        ++*(_DWORD *)(v4 + 2080);
        BUG();
      }
      v26 = v25 - 1;
      v7 = 0;
      v14 = v35;
      v27 = v26 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v28 = *(_QWORD *)(v4 + 2072);
      v29 = 1;
      v18 = *(_DWORD *)(v4 + 2080) + 1;
      v8 = (_QWORD *)(v28 + 16LL * v27);
      v30 = *v8;
      if ( v13 != *v8 )
      {
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v7 )
            v7 = (__int64)v8;
          v27 = v26 & (v29 + v27);
          v8 = (_QWORD *)(v28 + 16LL * v27);
          v30 = *v8;
          if ( v13 == *v8 )
            goto LABEL_9;
          ++v29;
        }
        if ( v7 )
          v8 = (_QWORD *)v7;
      }
    }
LABEL_9:
    *(_DWORD *)(v4 + 2080) = v18;
    if ( *v8 != -4096 )
      --*(_DWORD *)(v4 + 2084);
    *v8 = v13;
    *((_DWORD *)v8 + 2) = v14;
    v20 = *(unsigned int *)(v4 + 8);
    if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 12) )
    {
      sub_C8D5F0(v4, (const void *)(v4 + 16), v20 + 1, 8u, v14, v7);
      v20 = *(unsigned int *)(v4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v4 + 8 * v20) = v13;
    ++*(_DWORD *)(v4 + 8);
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v3 );
LABEL_14:
  v21 = (unsigned __int8 *)a2;
  v22 = a3;
  if ( a2 == a3 )
  {
    v31 = sub_ACADE0(*(__int64 ***)(a2 + 8));
    v21 = (unsigned __int8 *)a2;
    v22 = v31;
  }
  if ( !*(_QWORD *)(v22 + 16) && *(_BYTE *)v22 > 0x1Cu && (*(_BYTE *)(v22 + 7) & 0x10) == 0 && (v21[7] & 0x10) != 0 )
  {
    v37 = v21;
    v40 = v22;
    sub_BD6B90((unsigned __int8 *)v22, v21);
    v21 = v37;
    v22 = v40;
  }
  v39 = v21;
  sub_BD84D0((__int64)v21, v22);
  return v39;
}
