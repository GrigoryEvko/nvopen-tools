// Function: sub_13E8A40
// Address: 0x13e8a40
//
__int64 __fastcall sub_13E8A40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // r8d
  unsigned int v8; // edx
  __int64 *v9; // rbx
  __int64 v10; // rsi
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // r15
  __int64 v14; // rax
  unsigned int v15; // r8d
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // rsi
  int v24; // ecx
  int v25; // edi
  unsigned int v26; // eax
  __int64 v27; // rdx
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rcx
  int v31; // edx
  int v32; // r9d
  _QWORD *v33; // rdx

  v5 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a1 + 72);
    v7 = 1;
    v8 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v9 = (__int64 *)(v6 + 80LL * v8);
    v10 = *v9;
    if ( a3 == *v9 )
    {
LABEL_3:
      if ( v9 != (__int64 *)(v6 + 80 * v5) )
      {
        v11 = (_QWORD *)v9[3];
        v12 = (_QWORD *)v9[2];
        if ( v11 == v12 )
        {
          v13 = &v12[*((unsigned int *)v9 + 9)];
          if ( v12 == v13 )
          {
            v33 = (_QWORD *)v9[2];
          }
          else
          {
            do
            {
              if ( *v12 == a2 )
                break;
              ++v12;
            }
            while ( v13 != v12 );
            v33 = v13;
          }
LABEL_17:
          while ( v33 != v12 )
          {
            if ( *v12 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_8;
            ++v12;
          }
          v15 = 1;
          if ( v13 != v12 )
            return v15;
        }
        else
        {
          v13 = &v11[*((unsigned int *)v9 + 8)];
          v12 = (_QWORD *)sub_16CC9F0(v9 + 1, a2);
          if ( *v12 == a2 )
          {
            v29 = v9[3];
            if ( v29 == v9[2] )
              v30 = *((unsigned int *)v9 + 9);
            else
              v30 = *((unsigned int *)v9 + 8);
            v33 = (_QWORD *)(v29 + 8 * v30);
            goto LABEL_17;
          }
          v14 = v9[3];
          if ( v14 == v9[2] )
          {
            v12 = (_QWORD *)(v14 + 8LL * *((unsigned int *)v9 + 9));
            v33 = v12;
            goto LABEL_17;
          }
          v12 = (_QWORD *)(v14 + 8LL * *((unsigned int *)v9 + 8));
LABEL_8:
          v15 = 1;
          if ( v13 != v12 )
            return v15;
        }
      }
    }
    else
    {
      while ( v10 != -8 )
      {
        v8 = (v5 - 1) & (v7 + v8);
        v9 = (__int64 *)(v6 + 80LL * v8);
        v10 = *v9;
        if ( a3 == *v9 )
          goto LABEL_3;
        ++v7;
      }
    }
  }
  v17 = *(unsigned int *)(a1 + 56);
  if ( !(_DWORD)v17 )
    return 0;
  v18 = *(_QWORD *)(a1 + 40);
  v19 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (__int64 *)(v18 + 16LL * v19);
  v21 = *v20;
  if ( *v20 != a2 )
  {
    v31 = 1;
    while ( v21 != -8 )
    {
      v32 = v31 + 1;
      v19 = (v17 - 1) & (v31 + v19);
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( *v20 == a2 )
        goto LABEL_21;
      v31 = v32;
    }
    return 0;
  }
LABEL_21:
  if ( v20 == (__int64 *)(v18 + 16 * v17) )
    return 0;
  v22 = v20[1];
  if ( (*(_BYTE *)(v22 + 48) & 1) != 0 )
  {
    v23 = v22 + 56;
    v24 = 3;
  }
  else
  {
    v23 = *(_QWORD *)(v22 + 56);
    v28 = *(_DWORD *)(v22 + 64);
    v15 = 0;
    if ( !v28 )
      return v15;
    v24 = v28 - 1;
  }
  v15 = 1;
  v25 = 1;
  v26 = v24 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v27 = *(_QWORD *)(v23 + 48LL * v26);
  if ( a3 != v27 )
  {
    while ( v27 != -8 )
    {
      v26 = v24 & (v25 + v26);
      v27 = *(_QWORD *)(v23 + 48LL * v26);
      if ( a3 == v27 )
        return 1;
      ++v25;
    }
    return 0;
  }
  return v15;
}
