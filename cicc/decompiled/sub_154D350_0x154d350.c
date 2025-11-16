// Function: sub_154D350
// Address: 0x154d350
//
__int64 __fastcall sub_154D350(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rcx
  unsigned int v5; // esi
  unsigned int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 result; // rax
  int v11; // eax
  unsigned __int8 v12; // dl
  unsigned int v13; // eax
  int v14; // eax
  int v15; // r13d
  int v16; // eax
  unsigned int v17; // edx
  __int64 v18; // r8
  int v19; // eax
  int v20; // edx
  __int64 v21; // rax
  _QWORD *v22; // r13
  _QWORD *v23; // r14
  unsigned __int8 v24; // al
  int v25; // r9d
  int v26; // r10d
  __int64 v27; // r9
  __int64 v28; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v29[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a1;
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_DWORD *)(a2 + 24);
  if ( !v5 )
  {
    v12 = *(_BYTE *)(a1 + 16);
    if ( v12 > 0x10u || (v13 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 0 )
    {
      v19 = *(_DWORD *)(a2 + 16);
      v28 = a1;
      v15 = v19 + 1;
      goto LABEL_18;
    }
LABEL_12:
    if ( v12 > 3u )
    {
      v21 = 3LL * v13;
      if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
      {
        v22 = *(_QWORD **)(v3 - 8);
        v23 = &v22[v21];
      }
      else
      {
        v23 = (_QWORD *)v3;
        v22 = (_QWORD *)(v3 - v21 * 8);
      }
      do
      {
        v24 = *(_BYTE *)(*v22 + 16LL);
        if ( v24 != 18 && v24 > 3u )
          sub_154D350(*v22, a2);
        v22 += 3;
      }
      while ( v23 != v22 );
      v4 = *(_QWORD *)(a2 + 8);
      v5 = *(_DWORD *)(a2 + 24);
    }
    v16 = *(_DWORD *)(a2 + 16);
    v28 = v3;
    v15 = v16 + 1;
    if ( v5 )
    {
      v6 = v5 - 1;
LABEL_15:
      v17 = v6 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      result = v4 + 16LL * v17;
      v18 = *(_QWORD *)result;
      if ( v3 == *(_QWORD *)result )
      {
LABEL_16:
        *(_DWORD *)(result + 8) = v15;
        return result;
      }
      v26 = 1;
      v27 = 0;
      while ( v18 != -8 )
      {
        if ( v18 == -16 && !v27 )
          v27 = result;
        v17 = v6 & (v26 + v17);
        result = v4 + 16LL * v17;
        v18 = *(_QWORD *)result;
        if ( v3 == *(_QWORD *)result )
          goto LABEL_16;
        ++v26;
      }
      if ( v27 )
        result = v27;
      ++*(_QWORD *)a2;
      if ( 4 * v15 < 3 * v5 )
      {
        v20 = v15;
        if ( v5 - (v15 + *(_DWORD *)(a2 + 20)) > v5 >> 3 )
          goto LABEL_21;
        goto LABEL_20;
      }
LABEL_19:
      v5 *= 2;
LABEL_20:
      sub_1541430(a2, v5);
      sub_154C3E0(a2, &v28, v29);
      result = v29[0];
      v3 = v28;
      v20 = *(_DWORD *)(a2 + 16) + 1;
LABEL_21:
      *(_DWORD *)(a2 + 16) = v20;
      if ( *(_QWORD *)result != -8 )
        --*(_DWORD *)(a2 + 20);
      *(_QWORD *)result = v3;
      *(_DWORD *)(result + 8) = 0;
      *(_BYTE *)(result + 12) = 0;
      goto LABEL_16;
    }
LABEL_18:
    ++*(_QWORD *)a2;
    v5 = 0;
    goto LABEL_19;
  }
  v6 = v5 - 1;
  v7 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v8 = (__int64 *)(v4 + 16LL * v7);
  v9 = *v8;
  if ( v3 != *v8 )
  {
    v11 = 1;
    while ( v9 != -8 )
    {
      v25 = v11 + 1;
      v7 = v6 & (v11 + v7);
      v8 = (__int64 *)(v4 + 16LL * v7);
      v9 = *v8;
      if ( v3 == *v8 )
        goto LABEL_3;
      v11 = v25;
    }
LABEL_7:
    v12 = *(_BYTE *)(v3 + 16);
    if ( v12 > 0x10u || (v13 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF) == 0 )
    {
      v14 = *(_DWORD *)(a2 + 16);
      v28 = v3;
      v15 = v14 + 1;
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
