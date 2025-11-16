// Function: sub_267FA80
// Address: 0x267fa80
//
_QWORD *__fastcall sub_267FA80(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  unsigned int v4; // esi
  int v6; // r11d
  __int64 v7; // r8
  unsigned int v8; // edi
  _QWORD *v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rcx
  _QWORD *result; // rax
  _QWORD *v13; // rbx
  int v14; // ecx
  int v15; // ecx
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  volatile signed __int32 *v18; // r12
  signed __int32 v19; // eax
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // edx
  __int64 v24; // rdi
  int v25; // r10d
  _QWORD *v26; // r9
  int v27; // eax
  int v28; // edx
  __int64 v29; // rdi
  _QWORD *v30; // r8
  unsigned int v31; // r13d
  int v32; // r9d
  __int64 v33; // rsi
  signed __int32 v34; // eax

  v2 = a1 + 128;
  v4 = *(_DWORD *)(a1 + 152);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 128);
    goto LABEL_26;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 136);
  v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (_QWORD *)(v7 + 24LL * v8);
  v10 = 0;
  v11 = *v9;
  if ( *v9 == a2 )
  {
LABEL_3:
    result = (_QWORD *)v9[1];
    v13 = v9 + 1;
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v10 )
      v10 = v9;
    v8 = (v4 - 1) & (v6 + v8);
    v9 = (_QWORD *)(v7 + 24LL * v8);
    v11 = *v9;
    if ( *v9 == a2 )
      goto LABEL_3;
    ++v6;
  }
  v14 = *(_DWORD *)(a1 + 144);
  if ( !v10 )
    v10 = v9;
  ++*(_QWORD *)(a1 + 128);
  v15 = v14 + 1;
  if ( 4 * v15 >= 3 * v4 )
  {
LABEL_26:
    sub_267F7D0(v2, 2 * v4);
    v20 = *(_DWORD *)(a1 + 152);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 136);
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 144) + 1;
      v10 = (_QWORD *)(v22 + 24LL * v23);
      v24 = *v10;
      if ( *v10 != a2 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -4096 )
        {
          if ( !v26 && v24 == -8192 )
            v26 = v10;
          v23 = v21 & (v25 + v23);
          v10 = (_QWORD *)(v22 + 24LL * v23);
          v24 = *v10;
          if ( *v10 == a2 )
            goto LABEL_15;
          ++v25;
        }
        if ( v26 )
          v10 = v26;
      }
      goto LABEL_15;
    }
    goto LABEL_55;
  }
  if ( v4 - *(_DWORD *)(a1 + 148) - v15 <= v4 >> 3 )
  {
    sub_267F7D0(v2, v4);
    v27 = *(_DWORD *)(a1 + 152);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 136);
      v30 = 0;
      v31 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = 1;
      v15 = *(_DWORD *)(a1 + 144) + 1;
      v10 = (_QWORD *)(v29 + 24LL * v31);
      v33 = *v10;
      if ( *v10 != a2 )
      {
        while ( v33 != -4096 )
        {
          if ( !v30 && v33 == -8192 )
            v30 = v10;
          v31 = v28 & (v32 + v31);
          v10 = (_QWORD *)(v29 + 24LL * v31);
          v33 = *v10;
          if ( *v10 == a2 )
            goto LABEL_15;
          ++v32;
        }
        if ( v30 )
          v10 = v30;
      }
      goto LABEL_15;
    }
LABEL_55:
    ++*(_DWORD *)(a1 + 144);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 144) = v15;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 148);
  *v10 = a2;
  v13 = v10 + 1;
  v10[1] = 0;
  v10[2] = 0;
LABEL_18:
  v16 = (_QWORD *)sub_22077B0(0xA0u);
  v17 = v16;
  if ( v16 )
  {
    v16[1] = 0x100000001LL;
    *v16 = &unk_4A20640;
    v16[2] = v16 + 4;
    v16[3] = 0x1000000000LL;
  }
  v18 = (volatile signed __int32 *)v13[1];
  result = v16 + 2;
  v13[1] = v17;
  *v13 = v17 + 2;
  if ( v18 )
  {
    if ( &_pthread_key_create )
    {
      v19 = _InterlockedExchangeAdd(v18 + 2, 0xFFFFFFFF);
    }
    else
    {
      v19 = *((_DWORD *)v18 + 2);
      *((_DWORD *)v18 + 2) = v19 - 1;
    }
    if ( v19 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v18 + 16LL))(v18);
      if ( &_pthread_key_create )
      {
        v34 = _InterlockedExchangeAdd(v18 + 3, 0xFFFFFFFF);
      }
      else
      {
        v34 = *((_DWORD *)v18 + 3);
        *((_DWORD *)v18 + 3) = v34 - 1;
      }
      if ( v34 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v18 + 24LL))(v18);
    }
    return (_QWORD *)*v13;
  }
  return result;
}
