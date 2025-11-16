// Function: sub_1E03570
// Address: 0x1e03570
//
unsigned __int64 __fastcall sub_1E03570(__int64 *a1, __int64 a2)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // rbx
  unsigned __int64 result; // rax
  int v12; // r15d
  __int64 v13; // rbx
  char v14; // r9
  __int64 v15; // rax
  int v16; // ecx
  unsigned int v17; // esi
  int v18; // edx
  __int64 v19; // rdx
  unsigned __int64 *v20; // rax
  unsigned __int64 *v21; // rdx
  int v22; // r10d
  __int64 *v23; // r9
  int v24; // ecx
  int v25; // edx
  __int64 v26; // rdx
  __int64 v28; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *a1;
  v28 = a2;
  v5 = *(_DWORD *)(v4 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)v4;
    goto LABEL_39;
  }
  v6 = *(_QWORD *)(v4 + 8);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v22 = 1;
    v23 = 0;
    while ( v9 != -8 )
    {
      if ( !v23 && v9 == -16 )
        v23 = v8;
      v7 = (v5 - 1) & (v22 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      ++v22;
    }
    v24 = *(_DWORD *)(v4 + 16);
    if ( v23 )
      v8 = v23;
    ++*(_QWORD *)v4;
    v25 = v24 + 1;
    if ( 4 * (v24 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(v4 + 20) - v25 > v5 >> 3 )
      {
LABEL_35:
        *(_DWORD *)(v4 + 16) = v25;
        if ( *v8 != -8 )
          --*(_DWORD *)(v4 + 20);
        v26 = v28;
        *((_DWORD *)v8 + 2) = 0;
        *v8 = v26;
        goto LABEL_3;
      }
LABEL_40:
      sub_1E03290(v4, v5);
      sub_1DF9680(v4, &v28, v29);
      v8 = (__int64 *)v29[0];
      v25 = *(_DWORD *)(v4 + 16) + 1;
      goto LABEL_35;
    }
LABEL_39:
    v5 *= 2;
    goto LABEL_40;
  }
LABEL_3:
  *((_DWORD *)v8 + 2) = -1;
  v10 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v10 )
LABEL_48:
    BUG();
  result = *(_QWORD *)v10;
  if ( (*(_QWORD *)v10 & 4) == 0 && (*(_BYTE *)(v10 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      result &= 0xFFFFFFFFFFFFFFF8LL;
      v10 = result;
      if ( (*(_BYTE *)(result + 46) & 4) == 0 )
        break;
      result = *(_QWORD *)result;
    }
  }
  while ( a2 + 24 != v10 )
  {
    sub_21EA5F0(v29, a1[1], v10);
    if ( BYTE4(v29[0]) )
    {
      v12 = v29[0];
      if ( (unsigned int)(LODWORD(v29[0]) - 24) <= 0xE8 && (v29[0] & 7) == 0 )
      {
        v13 = a1[2];
        v28 = a2;
        v14 = sub_1DF9680(v13, &v28, v29);
        v15 = v29[0];
        if ( !v14 )
        {
          v16 = *(_DWORD *)(v13 + 16);
          v17 = *(_DWORD *)(v13 + 24);
          ++*(_QWORD *)v13;
          v18 = v16 + 1;
          if ( 4 * (v16 + 1) >= 3 * v17 )
          {
            v17 *= 2;
          }
          else if ( v17 - *(_DWORD *)(v13 + 20) - v18 > v17 >> 3 )
          {
            goto LABEL_12;
          }
          sub_1E03290(v13, v17);
          sub_1DF9680(v13, &v28, v29);
          v15 = v29[0];
          v18 = *(_DWORD *)(v13 + 16) + 1;
LABEL_12:
          *(_DWORD *)(v13 + 16) = v18;
          if ( *(_QWORD *)v15 != -8 )
            --*(_DWORD *)(v13 + 20);
          v19 = v28;
          *(_DWORD *)(v15 + 8) = 0;
          *(_QWORD *)v15 = v19;
        }
        *(_DWORD *)(v15 + 8) = v12;
        result = a1[1];
        *(_BYTE *)(result + 48) = 1;
        return result;
      }
    }
    v20 = (unsigned __int64 *)(*(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL);
    v21 = v20;
    if ( !v20 )
      goto LABEL_48;
    v10 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
    result = *v20;
    if ( (result & 4) == 0 && (*((_BYTE *)v21 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        result &= 0xFFFFFFFFFFFFFFF8LL;
        v10 = result;
        if ( (*(_BYTE *)(result + 46) & 4) == 0 )
          break;
        result = *(_QWORD *)result;
      }
    }
  }
  return result;
}
