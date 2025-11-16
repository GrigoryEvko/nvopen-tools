// Function: sub_15415F0
// Address: 0x15415f0
//
__int64 __fastcall sub_15415F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  unsigned int v5; // esi
  unsigned int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 result; // rax
  int v11; // eax
  unsigned __int8 v12; // dl
  unsigned int v13; // eax
  int v14; // r13d
  unsigned int v15; // r8d
  __int64 v16; // rdx
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // ecx
  int v21; // edx
  __int64 v22; // rdi
  __int64 v23; // rax
  _QWORD *v24; // r13
  _QWORD *v25; // r14
  unsigned __int8 v26; // al
  int v27; // r9d
  int v28; // r10d
  __int64 v29; // r9
  int v30; // eax
  int v31; // ecx
  __int64 v32; // rdi
  __int64 v33; // r8
  unsigned int v34; // r14d
  int v35; // r9d
  __int64 v36; // rsi
  int v37; // r10d
  __int64 v38; // r9

  v3 = *(_QWORD *)(a2 + 8);
  v5 = *(_DWORD *)(a2 + 24);
  if ( !v5 )
  {
    v12 = *(_BYTE *)(a1 + 16);
    if ( v12 > 0x10u || (v13 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 0 )
    {
      v14 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_18;
    }
LABEL_12:
    if ( v12 > 3u )
    {
      v23 = 3LL * v13;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      {
        v24 = *(_QWORD **)(a1 - 8);
        v25 = &v24[v23];
      }
      else
      {
        v25 = (_QWORD *)a1;
        v24 = (_QWORD *)(a1 - v23 * 8);
      }
      do
      {
        v26 = *(_BYTE *)(*v24 + 16LL);
        if ( v26 != 18 && v26 > 3u )
          sub_15415F0(*v24, a2);
        v24 += 3;
      }
      while ( v25 != v24 );
      v3 = *(_QWORD *)(a2 + 8);
      v5 = *(_DWORD *)(a2 + 24);
    }
    v14 = *(_DWORD *)(a2 + 16) + 1;
    if ( v5 )
    {
      v6 = v5 - 1;
LABEL_15:
      v15 = v6 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      result = v3 + 16LL * v15;
      v16 = *(_QWORD *)result;
      if ( a1 == *(_QWORD *)result )
      {
LABEL_16:
        *(_DWORD *)(result + 8) = v14;
        return result;
      }
      v28 = 1;
      v29 = 0;
      while ( v16 != -8 )
      {
        if ( !v29 && v16 == -16 )
          v29 = result;
        v15 = v6 & (v28 + v15);
        result = v3 + 16LL * v15;
        v16 = *(_QWORD *)result;
        if ( a1 == *(_QWORD *)result )
          goto LABEL_16;
        ++v28;
      }
      if ( v29 )
        result = v29;
      ++*(_QWORD *)a2;
      if ( 4 * v14 < 3 * v5 )
      {
        v21 = v14;
        if ( v5 - (v14 + *(_DWORD *)(a2 + 20)) > v5 >> 3 )
        {
LABEL_21:
          *(_DWORD *)(a2 + 16) = v21;
          if ( *(_QWORD *)result != -8 )
            --*(_DWORD *)(a2 + 20);
          *(_QWORD *)result = a1;
          *(_DWORD *)(result + 8) = 0;
          *(_BYTE *)(result + 12) = 0;
          goto LABEL_16;
        }
        sub_1541430(a2, v5);
        v30 = *(_DWORD *)(a2 + 24);
        if ( v30 )
        {
          v31 = v30 - 1;
          v32 = *(_QWORD *)(a2 + 8);
          v33 = 0;
          v34 = (v30 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v35 = 1;
          v21 = *(_DWORD *)(a2 + 16) + 1;
          result = v32 + 16LL * v34;
          v36 = *(_QWORD *)result;
          if ( a1 != *(_QWORD *)result )
          {
            while ( v36 != -8 )
            {
              if ( !v33 && v36 == -16 )
                v33 = result;
              v34 = v31 & (v35 + v34);
              result = v32 + 16LL * v34;
              v36 = *(_QWORD *)result;
              if ( a1 == *(_QWORD *)result )
                goto LABEL_21;
              ++v35;
            }
            if ( v33 )
              result = v33;
          }
          goto LABEL_21;
        }
LABEL_66:
        ++*(_DWORD *)(a2 + 16);
        BUG();
      }
LABEL_19:
      sub_1541430(a2, 2 * v5);
      v17 = *(_DWORD *)(a2 + 24);
      if ( v17 )
      {
        v18 = v17 - 1;
        v19 = *(_QWORD *)(a2 + 8);
        v20 = (v17 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v21 = *(_DWORD *)(a2 + 16) + 1;
        result = v19 + 16LL * v20;
        v22 = *(_QWORD *)result;
        if ( a1 != *(_QWORD *)result )
        {
          v37 = 1;
          v38 = 0;
          while ( v22 != -8 )
          {
            if ( !v38 && v22 == -16 )
              v38 = result;
            v20 = v18 & (v37 + v20);
            result = v19 + 16LL * v20;
            v22 = *(_QWORD *)result;
            if ( a1 == *(_QWORD *)result )
              goto LABEL_21;
            ++v37;
          }
          if ( v38 )
            result = v38;
        }
        goto LABEL_21;
      }
      goto LABEL_66;
    }
LABEL_18:
    ++*(_QWORD *)a2;
    v5 = 0;
    goto LABEL_19;
  }
  v6 = v5 - 1;
  v7 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v8 = (__int64 *)(v3 + 16LL * v7);
  v9 = *v8;
  if ( a1 != *v8 )
  {
    v11 = 1;
    while ( v9 != -8 )
    {
      v27 = v11 + 1;
      v7 = v6 & (v11 + v7);
      v8 = (__int64 *)(v3 + 16LL * v7);
      v9 = *v8;
      if ( a1 == *v8 )
        goto LABEL_3;
      v11 = v27;
    }
LABEL_7:
    v12 = *(_BYTE *)(a1 + 16);
    if ( v12 > 0x10u || (v13 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 0 )
    {
      v14 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_15;
    }
    goto LABEL_12;
  }
LABEL_3:
  result = *((unsigned int *)v8 + 2);
  if ( !(_DWORD)result )
    goto LABEL_7;
  return result;
}
