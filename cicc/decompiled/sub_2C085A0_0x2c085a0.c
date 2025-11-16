// Function: sub_2C085A0
// Address: 0x2c085a0
//
__int64 __fastcall sub_2C085A0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 result; // rax
  int v10; // edx
  unsigned int v11; // esi
  __int64 v12; // r8
  int v13; // r15d
  __int64 *v14; // r10
  unsigned int v15; // edi
  __int64 *v16; // rdx
  __int64 v17; // rcx
  __int64 *v18; // rdx
  int v19; // r9d
  int v20; // edi
  int v21; // ecx
  int v22; // edx
  int v23; // edx
  __int64 v24; // r8
  unsigned int v25; // esi
  __int64 v26; // rdi
  int v27; // r11d
  __int64 *v28; // r9
  int v29; // edx
  int v30; // esi
  __int64 v31; // rdi
  __int64 *v32; // r8
  unsigned int v33; // r14d
  int v34; // r9d
  __int64 v35; // rdx
  __int64 v36; // [rsp+8h] [rbp-38h]
  __int64 v37; // [rsp+8h] [rbp-38h]

  v4 = *(unsigned int *)(a1 + 96);
  v5 = *(_QWORD *)(a1 + 80);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return v7[1];
    }
    else
    {
      v10 = 1;
      while ( v8 != -4096 )
      {
        v19 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        v10 = v19;
      }
    }
  }
  result = sub_2AC42A0(*(_QWORD *)(a1 + 16), a2);
  v11 = *(_DWORD *)(a1 + 96);
  if ( !v11 )
  {
    ++*(_QWORD *)(a1 + 72);
    goto LABEL_27;
  }
  v12 = *(_QWORD *)(a1 + 80);
  v13 = 1;
  v14 = 0;
  v15 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v16 = (__int64 *)(v12 + 16LL * v15);
  v17 = *v16;
  if ( *v16 != a2 )
  {
    while ( v17 != -4096 )
    {
      if ( v17 == -8192 && !v14 )
        v14 = v16;
      v15 = (v11 - 1) & (v13 + v15);
      v16 = (__int64 *)(v12 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == a2 )
        goto LABEL_9;
      ++v13;
    }
    v20 = *(_DWORD *)(a1 + 88);
    if ( !v14 )
      v14 = v16;
    ++*(_QWORD *)(a1 + 72);
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v11 )
    {
      if ( v11 - *(_DWORD *)(a1 + 92) - v21 > v11 >> 3 )
      {
LABEL_23:
        *(_DWORD *)(a1 + 88) = v21;
        if ( *v14 != -4096 )
          --*(_DWORD *)(a1 + 92);
        *v14 = a2;
        v18 = v14 + 1;
        v14[1] = 0;
        goto LABEL_10;
      }
      v37 = result;
      sub_2AC40F0(a1 + 72, v11);
      v29 = *(_DWORD *)(a1 + 96);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a1 + 80);
        v32 = 0;
        v33 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v34 = 1;
        v21 = *(_DWORD *)(a1 + 88) + 1;
        result = v37;
        v14 = (__int64 *)(v31 + 16LL * v33);
        v35 = *v14;
        if ( *v14 != a2 )
        {
          while ( v35 != -4096 )
          {
            if ( !v32 && v35 == -8192 )
              v32 = v14;
            v33 = v30 & (v34 + v33);
            v14 = (__int64 *)(v31 + 16LL * v33);
            v35 = *v14;
            if ( *v14 == a2 )
              goto LABEL_23;
            ++v34;
          }
          if ( v32 )
            v14 = v32;
        }
        goto LABEL_23;
      }
LABEL_50:
      ++*(_DWORD *)(a1 + 88);
      BUG();
    }
LABEL_27:
    v36 = result;
    sub_2AC40F0(a1 + 72, 2 * v11);
    v22 = *(_DWORD *)(a1 + 96);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 80);
      v25 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = *(_DWORD *)(a1 + 88) + 1;
      result = v36;
      v14 = (__int64 *)(v24 + 16LL * v25);
      v26 = *v14;
      if ( *v14 != a2 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != -4096 )
        {
          if ( v26 == -8192 && !v28 )
            v28 = v14;
          v25 = v23 & (v27 + v25);
          v14 = (__int64 *)(v24 + 16LL * v25);
          v26 = *v14;
          if ( *v14 == a2 )
            goto LABEL_23;
          ++v27;
        }
        if ( v28 )
          v14 = v28;
      }
      goto LABEL_23;
    }
    goto LABEL_50;
  }
LABEL_9:
  v18 = v16 + 1;
LABEL_10:
  *v18 = result;
  return result;
}
