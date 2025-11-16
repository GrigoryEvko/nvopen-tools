// Function: sub_11C9DB0
// Address: 0x11c9db0
//
char *__fastcall sub_11C9DB0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        unsigned int *a7)
{
  unsigned __int8 v7; // al
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // ecx
  int *v13; // rdx
  int v14; // esi
  char *result; // rax
  __int64 v16; // rsi
  __int64 v17; // r8
  __int64 v19; // rsi
  int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // r8
  unsigned int v23; // ecx
  int v24; // esi
  int v25; // edx
  unsigned int v26; // ecx
  int v27; // esi
  int v28; // edx
  int v29; // edx
  int v30; // r10d
  int v31; // r10d
  int v32; // r10d

  v7 = *(_BYTE *)(a3 + 8);
  if ( v7 == 3 )
  {
    v17 = a4;
    *a7 = a4;
    if ( (a2[((unsigned __int64)a4 >> 6) + 1] & (1LL << a4)) != 0 )
      return 0;
    v19 = *a2;
    result = 0;
    v20 = ((int)*(unsigned __int8 *)(v19 + (a4 >> 2)) >> (2 * (a4 & 3))) & 3;
    if ( v20 )
    {
      if ( v20 == 3 )
        return (&off_4977320)[2 * v17];
      v21 = *(unsigned int *)(v19 + 160);
      v22 = *(_QWORD *)(v19 + 144);
      if ( (_DWORD)v21 )
      {
        v23 = (v21 - 1) & (37 * a4);
        v13 = (int *)(v22 + 40LL * v23);
        v24 = *v13;
        if ( a4 == *v13 )
          return (char *)*((_QWORD *)v13 + 1);
        v25 = 1;
        while ( v24 != -1 )
        {
          v31 = v25 + 1;
          v23 = (v21 - 1) & (v25 + v23);
          v13 = (int *)(v22 + 40LL * v23);
          v24 = *v13;
          if ( a4 == *v13 )
            return (char *)*((_QWORD *)v13 + 1);
          v25 = v31;
        }
      }
      v13 = (int *)(v22 + 40 * v21);
      return (char *)*((_QWORD *)v13 + 1);
    }
  }
  else
  {
    if ( v7 <= 3u )
    {
      if ( !v7 )
        BUG();
      if ( v7 == 2 )
      {
        v8 = a5;
        *a7 = a5;
        if ( (a2[((unsigned __int64)a5 >> 6) + 1] & (1LL << a5)) == 0 )
        {
          v9 = *a2;
          if ( (((int)*(unsigned __int8 *)(v9 + (a5 >> 2)) >> (2 * (a5 & 3))) & 3) != 0 )
          {
            if ( (((int)*(unsigned __int8 *)(v9 + (a5 >> 2)) >> (2 * (a5 & 3))) & 3) != 3 )
            {
              v10 = *(unsigned int *)(v9 + 160);
              v11 = *(_QWORD *)(v9 + 144);
              if ( (_DWORD)v10 )
              {
                v12 = (v10 - 1) & (37 * a5);
                v13 = (int *)(v11 + 40LL * v12);
                v14 = *v13;
                if ( a5 == *v13 )
                  return (char *)*((_QWORD *)v13 + 1);
                v29 = 1;
                while ( v14 != -1 )
                {
                  v30 = v29 + 1;
                  v12 = (v10 - 1) & (v29 + v12);
                  v13 = (int *)(v11 + 40LL * v12);
                  v14 = *v13;
                  if ( a5 == *v13 )
                    return (char *)*((_QWORD *)v13 + 1);
                  v29 = v30;
                }
              }
LABEL_30:
              v13 = (int *)(v11 + 40 * v10);
              return (char *)*((_QWORD *)v13 + 1);
            }
            return (&off_4977320)[2 * v8];
          }
        }
        return 0;
      }
    }
    v8 = a6;
    *a7 = a6;
    if ( (a2[((unsigned __int64)a6 >> 6) + 1] & (1LL << a6)) != 0 )
      return 0;
    v16 = *a2;
    result = 0;
    if ( (((int)*(unsigned __int8 *)(v16 + (a6 >> 2)) >> (2 * (a6 & 3))) & 3) != 0 )
    {
      if ( (((int)*(unsigned __int8 *)(v16 + (a6 >> 2)) >> (2 * (a6 & 3))) & 3) != 3 )
      {
        v10 = *(unsigned int *)(v16 + 160);
        v11 = *(_QWORD *)(v16 + 144);
        if ( (_DWORD)v10 )
        {
          v26 = (v10 - 1) & (37 * a6);
          v13 = (int *)(v11 + 40LL * v26);
          v27 = *v13;
          if ( a6 == *v13 )
            return (char *)*((_QWORD *)v13 + 1);
          v28 = 1;
          while ( v27 != -1 )
          {
            v32 = v28 + 1;
            v26 = (v10 - 1) & (v28 + v26);
            v13 = (int *)(v11 + 40LL * v26);
            v27 = *v13;
            if ( a6 == *v13 )
              return (char *)*((_QWORD *)v13 + 1);
            v28 = v32;
          }
        }
        goto LABEL_30;
      }
      return (&off_4977320)[2 * v8];
    }
  }
  return result;
}
