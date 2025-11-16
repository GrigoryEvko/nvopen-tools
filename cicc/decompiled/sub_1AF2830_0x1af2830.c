// Function: sub_1AF2830
// Address: 0x1af2830
//
__int64 __fastcall sub_1AF2830(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  bool v5; // zf
  __int64 v6; // rdi
  unsigned int v7; // r8d
  unsigned int v8; // r12d
  unsigned int v9; // r10d
  _QWORD *v10; // rcx
  __int64 v11; // r9
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r9
  int v15; // ecx
  int v16; // r8d
  __int64 v17; // r10
  unsigned int v18; // ecx
  int v19; // edi
  _QWORD *v20; // rsi
  __int64 v21; // r9
  int v22; // r13d
  int v23; // edi
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // r9
  _QWORD *v27; // r10
  unsigned int v28; // r12d
  int v29; // r11d
  __int64 v30; // r8
  int v31; // edx
  int v32; // r10d
  int v33; // r12d
  _QWORD *v34; // r11
  __int64 v35; // [rsp+0h] [rbp-30h]
  __int64 v36; // [rsp+0h] [rbp-30h]
  __int64 v37; // [rsp+8h] [rbp-28h]
  __int64 v38; // [rsp+8h] [rbp-28h]

  result = a1;
  v5 = *(_BYTE *)(a1 + 16) == 9;
  v6 = *(_QWORD *)(a3 + 8);
  v7 = *(_DWORD *)(a3 + 24);
  if ( !v5 )
  {
    if ( v7 )
    {
      v8 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
      v9 = (v7 - 1) & v8;
      v10 = (_QWORD *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        return result;
      v22 = 1;
      v20 = 0;
      while ( v11 != -8 )
      {
        if ( v20 || v11 != -16 )
          v10 = v20;
        v9 = (v7 - 1) & (v22 + v9);
        v11 = *(_QWORD *)(v6 + 16LL * v9);
        if ( a2 == v11 )
          return result;
        ++v22;
        v20 = v10;
        v10 = (_QWORD *)(v6 + 16LL * v9);
      }
      v23 = *(_DWORD *)(a3 + 16);
      if ( !v20 )
        v20 = v10;
      ++*(_QWORD *)a3;
      v19 = v23 + 1;
      if ( 4 * v19 < 3 * v7 )
      {
        if ( v7 - *(_DWORD *)(a3 + 20) - v19 > v7 >> 3 )
        {
LABEL_12:
          *(_DWORD *)(a3 + 16) = v19;
          if ( *v20 != -8 )
            --*(_DWORD *)(a3 + 20);
          *v20 = a2;
          v20[1] = result;
          return result;
        }
        v38 = a3;
        v36 = result;
        sub_141A900(a3, v7);
        a3 = v38;
        v24 = *(_DWORD *)(v38 + 24);
        if ( v24 )
        {
          v25 = v24 - 1;
          v26 = *(_QWORD *)(v38 + 8);
          v27 = 0;
          v28 = v25 & v8;
          v29 = 1;
          v19 = *(_DWORD *)(v38 + 16) + 1;
          result = v36;
          v20 = (_QWORD *)(v26 + 16LL * v28);
          v30 = *v20;
          if ( a2 != *v20 )
          {
            while ( v30 != -8 )
            {
              if ( !v27 && v30 == -16 )
                v27 = v20;
              v28 = v25 & (v29 + v28);
              v20 = (_QWORD *)(v26 + 16LL * v28);
              v30 = *v20;
              if ( a2 == *v20 )
                goto LABEL_12;
              ++v29;
            }
            if ( v27 )
              v20 = v27;
          }
          goto LABEL_12;
        }
LABEL_51:
        ++*(_DWORD *)(a3 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    v37 = a3;
    v35 = result;
    sub_141A900(a3, 2 * v7);
    a3 = v37;
    v15 = *(_DWORD *)(v37 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(v37 + 8);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(v37 + 16) + 1;
      result = v35;
      v20 = (_QWORD *)(v17 + 16LL * v18);
      v21 = *v20;
      if ( a2 != *v20 )
      {
        v33 = 1;
        v34 = 0;
        while ( v21 != -8 )
        {
          if ( !v34 && v21 == -16 )
            v34 = v20;
          v18 = v16 & (v33 + v18);
          v20 = (_QWORD *)(v17 + 16LL * v18);
          v21 = *v20;
          if ( a2 == *v20 )
            goto LABEL_12;
          ++v33;
        }
        if ( v34 )
          v20 = v34;
      }
      goto LABEL_12;
    }
    goto LABEL_51;
  }
  if ( v7 )
  {
    v12 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v6 + 16LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
    {
LABEL_7:
      if ( v13 != (__int64 *)(v6 + 16LL * v7) )
        return v13[1];
    }
    else
    {
      v31 = 1;
      while ( v14 != -8 )
      {
        v32 = v31 + 1;
        v12 = (v7 - 1) & (v31 + v12);
        v13 = (__int64 *)(v6 + 16LL * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          goto LABEL_7;
        v31 = v32;
      }
    }
  }
  return result;
}
