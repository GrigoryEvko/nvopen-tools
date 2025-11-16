// Function: sub_18FE3A0
// Address: 0x18fe3a0
//
__int64 __fastcall sub_18FE3A0(__int64 *a1)
{
  __int64 result; // rax
  __int64 *i; // r15
  __int64 v3; // r13
  __int64 *v4; // r12
  unsigned int v5; // ebx
  int v6; // ebx
  __int64 *v7; // r9
  int v8; // eax
  __int64 v9; // rax
  unsigned int v10; // ebx
  char v11; // al
  __int64 *v12; // rbx
  int v13; // eax
  int v14; // ebx
  int v15; // ebx
  char v16; // al
  int v17; // ebx
  char v18; // al
  int v19; // [rsp+Ch] [rbp-54h]
  int v20; // [rsp+Ch] [rbp-54h]
  int v21; // [rsp+Ch] [rbp-54h]
  __int64 *v22; // [rsp+10h] [rbp-50h]
  unsigned int v23; // [rsp+10h] [rbp-50h]
  __int64 *v24; // [rsp+10h] [rbp-50h]
  __int64 *v25; // [rsp+18h] [rbp-48h]
  int v26; // [rsp+18h] [rbp-48h]
  __int64 *v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  unsigned int v29; // [rsp+20h] [rbp-40h]
  unsigned int v30; // [rsp+20h] [rbp-40h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 *v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]
  unsigned int v34; // [rsp+28h] [rbp-38h]
  unsigned int v35; // [rsp+28h] [rbp-38h]

  result = *a1;
  *(_QWORD *)(*a1 + 32) = a1[1];
  for ( i = (__int64 *)a1[2]; i; i = (__int64 *)a1[2] )
  {
    v3 = *a1;
    v4 = i + 2;
    v5 = *(_DWORD *)(*a1 + 24);
    v33 = *(_QWORD *)(*a1 + 8);
    if ( i[1] )
    {
      if ( v5 )
      {
        v10 = v5 - 1;
        v19 = 1;
        v22 = 0;
        v29 = v10 & sub_18FDEE0(*v4);
        while ( 1 )
        {
          v25 = (__int64 *)(v33 + 16LL * v29);
          v11 = sub_18FB980(*v4, *v25);
          v7 = v25;
          if ( v11 )
            break;
          if ( *v25 == -8 )
            goto LABEL_24;
          if ( *v25 == -16 )
          {
            if ( *v25 == -8 )
            {
LABEL_24:
              v5 = *(_DWORD *)(v3 + 24);
              if ( v22 )
                v7 = v22;
              v13 = *(_DWORD *)(v3 + 16);
              ++*(_QWORD *)v3;
              v8 = v13 + 1;
              if ( 4 * v8 >= 3 * v5 )
                goto LABEL_5;
              if ( v5 - (v8 + *(_DWORD *)(v3 + 20)) > v5 >> 3 )
                goto LABEL_7;
              sub_18FE1A0(v3, v5);
              v14 = *(_DWORD *)(v3 + 24);
              v7 = 0;
              if ( v14 )
              {
                v15 = v14 - 1;
                v31 = *(_QWORD *)(v3 + 8);
                v20 = 1;
                v24 = 0;
                v34 = v15 & sub_18FDEE0(*v4);
                while ( 1 )
                {
                  v27 = (__int64 *)(v31 + 16LL * v34);
                  v16 = sub_18FB980(*v4, *v27);
                  v7 = v27;
                  if ( v16 )
                    break;
                  if ( *v27 == -8 )
                    goto LABEL_41;
                  if ( *v27 == -16 )
                  {
                    if ( *v27 == -8 )
                    {
LABEL_41:
                      if ( v24 )
                        v7 = v24;
                      goto LABEL_6;
                    }
                    if ( !v24 )
                    {
                      if ( *v27 != -16 )
                        v7 = 0;
                      v24 = v7;
                    }
                  }
                  v34 = v15 & (v20 + v34);
                  ++v20;
                }
              }
              goto LABEL_6;
            }
            if ( !v22 )
            {
              if ( *v25 != -16 )
                v7 = 0;
              v22 = v7;
            }
          }
          v29 = v10 & (v19 + v29);
          ++v19;
        }
      }
      else
      {
        ++*(_QWORD *)v3;
LABEL_5:
        sub_18FE1A0(v3, 2 * v5);
        v6 = *(_DWORD *)(v3 + 24);
        v7 = 0;
        if ( v6 )
        {
          v17 = v6 - 1;
          v28 = *(_QWORD *)(v3 + 8);
          v21 = 1;
          v24 = 0;
          v35 = v17 & sub_18FDEE0(*v4);
          while ( 1 )
          {
            v32 = (__int64 *)(v28 + 16LL * v35);
            v18 = sub_18FB980(*v4, *v32);
            v7 = v32;
            if ( v18 )
              break;
            if ( *v32 == -8 )
              goto LABEL_41;
            if ( *v32 == -16 )
            {
              if ( *v32 == -8 )
                goto LABEL_41;
              if ( !v24 )
              {
                if ( *v32 != -16 )
                  v7 = 0;
                v24 = v7;
              }
            }
            v35 = v17 & (v21 + v35);
            ++v21;
          }
        }
LABEL_6:
        v8 = *(_DWORD *)(v3 + 16) + 1;
LABEL_7:
        *(_DWORD *)(v3 + 16) = v8;
        if ( *v7 != -8 )
          --*(_DWORD *)(v3 + 20);
        v9 = i[2];
        v7[1] = 0;
        *v7 = v9;
      }
      v7[1] = i[1];
LABEL_13:
      v3 = *a1;
    }
    else if ( v5 )
    {
      v26 = 1;
      v23 = v5 - 1;
      v30 = (v5 - 1) & sub_18FDEE0(*v4);
      while ( 1 )
      {
        v12 = (__int64 *)(v33 + 16LL * v30);
        if ( (unsigned __int8)sub_18FB980(*v4, *v12) )
          break;
        if ( *v12 == -8 )
          goto LABEL_13;
        v30 = v23 & (v26 + v30);
        ++v26;
      }
      *v12 = -16;
      --*(_DWORD *)(v3 + 16);
      ++*(_DWORD *)(v3 + 20);
      v3 = *a1;
    }
    a1[2] = *i;
    result = *(_QWORD *)(v3 + 40);
    *i = result;
    *(_QWORD *)(v3 + 40) = i;
  }
  return result;
}
