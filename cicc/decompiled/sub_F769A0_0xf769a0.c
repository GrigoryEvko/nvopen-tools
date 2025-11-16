// Function: sub_F769A0
// Address: 0xf769a0
//
signed __int64 __fastcall sub_F769A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  signed __int64 result; // rax
  __int64 v7; // r13
  __int64 v9; // rdx
  const void *v10; // r12
  unsigned __int64 v11; // r8
  __int64 v12; // r14
  __int64 i; // r12
  __int64 v14; // r8
  __int64 *v15; // rdi
  __int64 v16; // r13
  char v17; // cl
  __int64 v18; // r9
  int v19; // esi
  unsigned int v20; // edx
  __int64 v21; // r11
  __int64 v22; // rdx
  unsigned int v23; // esi
  unsigned int v24; // eax
  _QWORD *v25; // r10
  int v26; // edx
  int v27; // r15d
  __int64 v28; // rcx
  int v29; // edx
  unsigned int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // rcx
  int v33; // edx
  unsigned int v34; // eax
  __int64 v35; // rsi
  int v36; // r8d
  _QWORD *v37; // rdi
  int v38; // eax
  int v39; // edx
  int v40; // r8d
  __int64 v41; // [rsp-40h] [rbp-40h]

  result = 8LL * *(unsigned int *)(a2 + 8);
  if ( result )
  {
    v7 = result >> 3;
    v9 = *(unsigned int *)(a1 + 88);
    v10 = *(const void **)a2;
    v11 = v9 + (result >> 3);
    v12 = v9;
    if ( v11 > *(unsigned int *)(a1 + 92) )
    {
      v41 = 8LL * *(unsigned int *)(a2 + 8);
      sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v9 + (result >> 3), 8u, v11, a6);
      v9 = *(unsigned int *)(a1 + 88);
      result = v41;
    }
    result = (signed __int64)memcpy((void *)(*(_QWORD *)(a1 + 80) + 8 * v9), v10, result);
    LODWORD(i) = v7 + *(_DWORD *)(a1 + 88);
    *(_DWORD *)(a1 + 88) = i;
    for ( i = (unsigned int)i; ; v25[1] = i )
    {
LABEL_5:
      if ( --i < v12 )
        return result;
      while ( 1 )
      {
        v14 = *(_QWORD *)(a1 + 80);
        v15 = (__int64 *)(v14 + 8 * i);
        v16 = *v15;
        v17 = *(_BYTE *)(a1 + 8) & 1;
        if ( v17 )
        {
          v18 = a1 + 16;
          v19 = 3;
        }
        else
        {
          v23 = *(_DWORD *)(a1 + 24);
          v18 = *(_QWORD *)(a1 + 16);
          if ( !v23 )
          {
            v24 = *(_DWORD *)(a1 + 8);
            ++*(_QWORD *)a1;
            v25 = 0;
            v26 = (v24 >> 1) + 1;
            goto LABEL_16;
          }
          v19 = v23 - 1;
        }
        v20 = v19 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        result = v18 + 16LL * v20;
        v21 = *(_QWORD *)result;
        if ( v16 != *(_QWORD *)result )
          break;
LABEL_9:
        v22 = *(_QWORD *)(result + 8);
        if ( v22 < v12 )
        {
          *(_QWORD *)(v14 + 8 * v22) = 0;
          *(_QWORD *)(result + 8) = i;
          goto LABEL_5;
        }
        --i;
        *v15 = 0;
        if ( i < v12 )
          return result;
      }
      v27 = 1;
      v25 = 0;
      while ( v21 != -4096 )
      {
        if ( v21 == -8192 && !v25 )
          v25 = (_QWORD *)result;
        v20 = v19 & (v27 + v20);
        result = v18 + 16LL * v20;
        v21 = *(_QWORD *)result;
        if ( v16 == *(_QWORD *)result )
          goto LABEL_9;
        ++v27;
      }
      if ( !v25 )
        v25 = (_QWORD *)result;
      v24 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v26 = (v24 >> 1) + 1;
      if ( v17 )
      {
        v23 = 4;
        if ( (unsigned int)(4 * v26) >= 0xC )
          goto LABEL_27;
      }
      else
      {
        v23 = *(_DWORD *)(a1 + 24);
LABEL_16:
        if ( 3 * v23 <= 4 * v26 )
        {
LABEL_27:
          sub_F76580(a1, 2 * v23);
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v28 = a1 + 16;
            v29 = 3;
          }
          else
          {
            v38 = *(_DWORD *)(a1 + 24);
            v28 = *(_QWORD *)(a1 + 16);
            if ( !v38 )
              goto LABEL_61;
            v29 = v38 - 1;
          }
          v30 = v29 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v25 = (_QWORD *)(v28 + 16LL * v30);
          v31 = *v25;
          if ( v16 == *v25 )
            goto LABEL_30;
          v40 = 1;
          v37 = 0;
          while ( v31 != -4096 )
          {
            if ( !v37 && v31 == -8192 )
              v37 = v25;
            v30 = v29 & (v40 + v30);
            v25 = (_QWORD *)(v28 + 16LL * v30);
            v31 = *v25;
            if ( v16 == *v25 )
              goto LABEL_30;
            ++v40;
          }
          goto LABEL_37;
        }
      }
      if ( v23 - *(_DWORD *)(a1 + 12) - v26 > v23 >> 3 )
        goto LABEL_18;
      sub_F76580(a1, v23);
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v32 = a1 + 16;
        v33 = 3;
      }
      else
      {
        v39 = *(_DWORD *)(a1 + 24);
        v32 = *(_QWORD *)(a1 + 16);
        if ( !v39 )
        {
LABEL_61:
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          BUG();
        }
        v33 = v39 - 1;
      }
      v34 = v33 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v25 = (_QWORD *)(v32 + 16LL * v34);
      v35 = *v25;
      if ( v16 != *v25 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v37 )
            v37 = v25;
          v34 = v33 & (v36 + v34);
          v25 = (_QWORD *)(v32 + 16LL * v34);
          v35 = *v25;
          if ( v16 == *v25 )
            goto LABEL_30;
          ++v36;
        }
LABEL_37:
        if ( v37 )
          v25 = v37;
      }
LABEL_30:
      v24 = *(_DWORD *)(a1 + 8);
LABEL_18:
      result = (2 * (v24 >> 1) + 2) | v24 & 1;
      *(_DWORD *)(a1 + 8) = result;
      if ( *v25 != -4096 )
        --*(_DWORD *)(a1 + 12);
      *v25 = v16;
    }
  }
  return result;
}
