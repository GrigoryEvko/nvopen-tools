// Function: sub_D9A290
// Address: 0xd9a290
//
__int64 __fastcall sub_D9A290(__int64 a1, unsigned __int64 a2, unsigned __int8 a3)
{
  __int64 v3; // rax
  __int64 v4; // r9
  __int64 result; // rax
  __int64 v6; // r8
  unsigned int v7; // ecx
  __int64 *v8; // r14
  __int64 v9; // rdi
  __int64 v10; // r12
  __int64 v11; // r9
  _QWORD *v12; // r10
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // rbx
  bool v17; // zf
  __int64 v18; // rcx
  __int64 v19; // rsi
  unsigned int v20; // edx
  __int64 *v21; // r14
  __int64 v22; // rdi
  _QWORD *v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rbx
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 *v29; // rax
  int v30; // r14d
  int i; // r11d
  __int64 *v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  _QWORD *v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+28h] [rbp-48h]
  int v37; // [rsp+28h] [rbp-48h]
  _QWORD v38[2]; // [rsp+30h] [rbp-40h] BYREF
  _BYTE v39[48]; // [rsp+40h] [rbp-30h] BYREF

  v3 = a1 + 680;
  if ( !a3 )
    v3 = a1 + 648;
  v33 = v3;
  v4 = *(_QWORD *)(v3 + 8);
  result = *(unsigned int *)(v3 + 24);
  if ( !(_DWORD)result )
    return result;
  v6 = a1;
  v7 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v4 + 168LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    for ( i = 1; ; ++i )
    {
      if ( v9 == -4096 )
        return result;
      v7 = (result - 1) & (i + v7);
      v8 = (__int64 *)(v4 + 168LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        break;
    }
  }
  result = v4 + 168 * result;
  if ( v8 == (__int64 *)result )
    return result;
  v10 = v8[1];
  v11 = v10 + 112LL * *((unsigned int *)v8 + 4);
  if ( v11 == v10 )
    goto LABEL_30;
  v32 = v8;
  v12 = v38;
  a2 = (4LL * a3) | a2 & 0xFFFFFFFFFFFFFFFBLL;
  v13 = a2;
  do
  {
    v14 = *(_QWORD *)(v10 + 40);
    v15 = *(_QWORD *)(v10 + 56);
    v16 = v12;
    v17 = *(_WORD *)(v14 + 24) == 0;
    v38[0] = v14;
    v38[1] = v15;
    if ( v17 )
      goto LABEL_9;
    while ( 1 )
    {
      v18 = *(unsigned int *)(v6 + 736);
      v19 = *(_QWORD *)(v6 + 720);
      if ( (_DWORD)v18 )
      {
        v20 = (v18 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v21 = (__int64 *)(v19 + 72LL * v20);
        v22 = *v21;
        if ( v14 == *v21 )
          goto LABEL_13;
        v30 = 1;
        while ( v22 != -4096 )
        {
          v20 = (v18 - 1) & (v30 + v20);
          v37 = v30 + 1;
          v21 = (__int64 *)(v19 + 72LL * v20);
          v22 = *v21;
          if ( v14 == *v21 )
            goto LABEL_13;
          v30 = v37;
        }
      }
      v21 = (__int64 *)(v19 + 72 * v18);
LABEL_13:
      if ( !*((_BYTE *)v21 + 36) )
      {
        a2 = v13;
        v34 = v12;
        v35 = v6;
        v36 = v11;
        v29 = sub_C8CA60((__int64)(v21 + 1), v13);
        v11 = v36;
        v6 = v35;
        v12 = v34;
        if ( v29 )
        {
          *v29 = -2;
          ++*((_DWORD *)v21 + 8);
          ++v21[1];
        }
LABEL_9:
        v16 += 8;
        if ( v39 == v16 )
          break;
        goto LABEL_10;
      }
      a2 = v21[2];
      v23 = (_QWORD *)(a2 + 8LL * *((unsigned int *)v21 + 7));
      v24 = (_QWORD *)a2;
      if ( (_QWORD *)a2 == v23 )
        goto LABEL_9;
      while ( v13 != *v24 )
      {
        if ( v23 == ++v24 )
          goto LABEL_9;
      }
      v25 = (unsigned int)(*((_DWORD *)v21 + 7) - 1);
      v16 += 8;
      *((_DWORD *)v21 + 7) = v25;
      *v24 = *(_QWORD *)(a2 + 8 * v25);
      ++v21[1];
      if ( v39 == v16 )
        break;
LABEL_10:
      v14 = *(_QWORD *)v16;
      if ( !*(_WORD *)(*(_QWORD *)v16 + 24LL) )
        goto LABEL_9;
    }
    v10 += 112;
  }
  while ( v10 != v11 );
  v8 = v32;
  v26 = v32[1];
  v10 = v26 + 112LL * *((unsigned int *)v32 + 4);
  if ( v26 != v10 )
  {
    do
    {
      v10 -= 112;
      v27 = *(_QWORD *)(v10 + 64);
      if ( v27 != v10 + 80 )
        _libc_free(v27, a2);
      if ( *(_BYTE *)(v10 + 32) )
        *(_QWORD *)(v10 + 24) = 0;
      v28 = *(_QWORD *)(v10 + 24);
      *(_QWORD *)v10 = &unk_49DB368;
      if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
        sub_BD60C0((_QWORD *)(v10 + 8));
    }
    while ( v26 != v10 );
    v10 = v32[1];
  }
LABEL_30:
  if ( (__int64 *)v10 != v8 + 3 )
    _libc_free(v10, a2);
  result = v33;
  *v8 = -8192;
  --*(_DWORD *)(v33 + 16);
  ++*(_DWORD *)(v33 + 20);
  return result;
}
