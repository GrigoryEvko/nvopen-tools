// Function: sub_3599670
// Address: 0x3599670
//
__int64 __fastcall sub_3599670(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v4; // rdi
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r9
  int v11; // r12d
  int v12; // eax
  int v13; // r8d
  int v14; // ebx
  int v15; // ecx
  __int64 v16; // rsi
  char v17; // r10
  int v18; // r9d
  unsigned int i; // eax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v24; // rax
  __int64 v25; // r8
  unsigned int v26; // ecx
  __int64 *v27; // rdx
  __int64 v28; // r10
  int v29; // r13d
  int v30; // eax
  int v31; // edx
  int v32; // edx
  int v33; // r11d
  int v34; // r10d

  LOBYTE(v2) = *(_WORD *)(a2 + 68) == 68 || *(_WORD *)(a2 + 68) == 0;
  if ( !(_BYTE)v2 )
    return v2;
  v4 = *a1;
  v6 = *(unsigned int *)(v4 + 56);
  v7 = *(_QWORD *)(v4 + 40);
  if ( (_DWORD)v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_4:
      if ( v9 != (__int64 *)(v7 + 16 * v6) )
      {
        v11 = *((_DWORD *)v9 + 2);
        goto LABEL_6;
      }
    }
    else
    {
      v31 = 1;
      while ( v10 != -4096 )
      {
        v34 = v31 + 1;
        v8 = (v6 - 1) & (v31 + v8);
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_4;
        v31 = v34;
      }
    }
  }
  v11 = -1;
LABEL_6:
  v12 = sub_3598DB0(v4, a2);
  v13 = 0;
  v14 = v12;
  v15 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  if ( v15 != 1 )
  {
    v16 = *(_QWORD *)(a2 + 32);
    v17 = 0;
    v18 = 0;
    for ( i = 1; i != v15; i += 2 )
    {
      while ( *(_QWORD *)(a2 + 24) != *(_QWORD *)(v16 + 40LL * (i + 1) + 24) )
      {
        i += 2;
        if ( v15 == i )
          goto LABEL_11;
      }
      v20 = i;
      v17 = v2;
      v18 = *(_DWORD *)(v16 + 40 * v20 + 8);
    }
LABEL_11:
    if ( v17 )
      v13 = v18;
  }
  v21 = sub_2EBEE10(a1[3], v13);
  v22 = v21;
  if ( v21 )
  {
    LOBYTE(v2) = *(_WORD *)(v21 + 68) == 68 || *(_WORD *)(v21 + 68) == 0;
    if ( !(_BYTE)v2 )
    {
      v24 = *(unsigned int *)(*a1 + 56);
      v25 = *(_QWORD *)(*a1 + 40);
      if ( (_DWORD)v24 )
      {
        v26 = (v24 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v27 = (__int64 *)(v25 + 16LL * v26);
        v28 = *v27;
        if ( v22 == *v27 )
        {
LABEL_18:
          if ( v27 != (__int64 *)(v25 + 16 * v24) )
          {
            v29 = *((_DWORD *)v27 + 2);
LABEL_20:
            v30 = sub_3598DB0(*a1, v22);
            LOBYTE(v29) = v11 < v29;
            LOBYTE(v30) = v14 >= v30;
            return v30 | (unsigned int)v29;
          }
        }
        else
        {
          v32 = 1;
          while ( v28 != -4096 )
          {
            v33 = v32 + 1;
            v26 = (v24 - 1) & (v32 + v26);
            v27 = (__int64 *)(v25 + 16LL * v26);
            v28 = *v27;
            if ( v22 == *v27 )
              goto LABEL_18;
            v32 = v33;
          }
        }
      }
      v29 = -1;
      goto LABEL_20;
    }
  }
  return v2;
}
