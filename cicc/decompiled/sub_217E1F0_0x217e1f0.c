// Function: sub_217E1F0
// Address: 0x217e1f0
//
__int64 __fastcall sub_217E1F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v6; // r13d
  __int64 v7; // rbx
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r9
  unsigned int v11; // esi
  unsigned int v12; // r10d
  __int64 v13; // rcx
  unsigned int v14; // edx
  __int64 *v15; // rdi
  __int64 v16; // r8
  unsigned int v17; // esi
  __int64 *v18; // rdx
  __int64 v19; // rcx
  __int64 *v20; // rdx
  __int64 v21; // rcx
  int v22; // edi
  int v23; // edx
  int v24; // edi
  unsigned int v25; // [rsp+8h] [rbp-48h]
  int v26; // [rsp+Ch] [rbp-44h]
  int v28; // [rsp+18h] [rbp-38h]
  unsigned int v29; // [rsp+1Ch] [rbp-34h]

  result = *(unsigned int *)(a3 + 8);
  v28 = result;
  if ( (unsigned int)result > 1 )
  {
    v29 = 0;
    v25 = result - 1;
    while ( 1 )
    {
      result = v29;
      v6 = v29;
      if ( v28 != v29 )
        break;
LABEL_26:
      if ( v29 >= v25 )
        return result;
    }
    while ( 1 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)a3 + 8LL * v6);
      v8 = sub_217DB60(v7);
      v9 = *(_QWORD *)(a1 + 248);
      if ( v8 < 0 )
        result = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 16LL * (v8 & 0x7FFFFFFF) + 8);
      else
        result = *(_QWORD *)(*(_QWORD *)(v9 + 272) + 8LL * (unsigned int)v8);
      v10 = *(_QWORD *)(a2 + 8);
      v11 = *(_DWORD *)(a2 + 24);
      if ( !result )
        goto LABEL_19;
      while ( (*(_BYTE *)(result + 3) & 0x10) != 0 || (*(_BYTE *)(result + 4) & 8) != 0 )
      {
        result = *(_QWORD *)(result + 32);
        if ( !result )
          goto LABEL_19;
      }
      v12 = v11 - 1;
LABEL_12:
      if ( !v11 )
        goto LABEL_15;
      v13 = *(_QWORD *)(result + 16);
      v14 = v12 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v15 = (__int64 *)(v10 + 8LL * v14);
      v16 = *v15;
      if ( v13 != *v15 )
        break;
LABEL_14:
      if ( (__int64 *)(v10 + 8LL * v11) == v15 )
        goto LABEL_15;
LABEL_25:
      if ( ++v6 == v28 )
        goto LABEL_26;
    }
    v22 = 1;
    while ( v16 != -8 )
    {
      v14 = v12 & (v22 + v14);
      v26 = v22 + 1;
      v15 = (__int64 *)(v10 + 8LL * v14);
      v16 = *v15;
      if ( v13 == *v15 )
        goto LABEL_14;
      v22 = v26;
    }
LABEL_15:
    while ( 1 )
    {
      result = *(_QWORD *)(result + 32);
      if ( !result )
        break;
      while ( (*(_BYTE *)(result + 3) & 0x10) == 0 )
      {
        if ( (*(_BYTE *)(result + 4) & 8) == 0 )
          goto LABEL_12;
        result = *(_QWORD *)(result + 32);
        if ( !result )
          goto LABEL_19;
      }
    }
LABEL_19:
    if ( v11 )
    {
      v17 = v11 - 1;
      result = v17 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v18 = (__int64 *)(v10 + 8 * result);
      v19 = *v18;
      if ( v7 == *v18 )
      {
LABEL_21:
        *v18 = -16;
        --*(_DWORD *)(a2 + 16);
        ++*(_DWORD *)(a2 + 20);
      }
      else
      {
        v23 = 1;
        while ( v19 != -8 )
        {
          v24 = v23 + 1;
          result = v17 & (v23 + (_DWORD)result);
          v18 = (__int64 *)(v10 + 8LL * (unsigned int)result);
          v19 = *v18;
          if ( v7 == *v18 )
            goto LABEL_21;
          v23 = v24;
        }
      }
    }
    if ( v6 != v29 )
    {
      result = *(_QWORD *)a3 + 8LL * v6;
      v20 = (__int64 *)(*(_QWORD *)a3 + 8LL * v29);
      v21 = *v20;
      *v20 = *(_QWORD *)result;
      *(_QWORD *)result = v21;
    }
    ++v29;
    goto LABEL_25;
  }
  return result;
}
