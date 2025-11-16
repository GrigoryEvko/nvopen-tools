// Function: sub_8E8A30
// Address: 0x8e8a30
//
unsigned __int8 *__fastcall sub_8E8A30(unsigned __int8 *a1, char a2, char a3, char a4, int a5, __int64 a6)
{
  unsigned __int8 *v6; // r10
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  unsigned __int8 v12; // al
  int v13; // ebx
  int v14; // edx
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rcx
  unsigned __int8 *v19; // r10
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rcx
  unsigned __int8 *v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rcx
  unsigned __int8 *v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rax

  v6 = a1;
  if ( *(_QWORD *)(a6 + 32) )
  {
    v12 = *a1;
    if ( a2 == *a1 )
      return v6;
    goto LABEL_7;
  }
  v9 = *(_QWORD *)(a6 + 8);
  v10 = v9 + 1;
  if ( *(_DWORD *)(a6 + 28) )
  {
LABEL_6:
    *(_QWORD *)(a6 + 8) = v10;
    v12 = *v6;
    if ( a2 != *v6 )
      goto LABEL_7;
    goto LABEL_46;
  }
  v11 = *(_QWORD *)(a6 + 16);
  if ( v11 <= v10 )
  {
    *(_DWORD *)(a6 + 28) = 1;
    if ( v11 )
    {
      *(_BYTE *)(*(_QWORD *)a6 + v11 - 1) = 0;
      v10 = *(_QWORD *)(a6 + 8) + 1LL;
    }
    goto LABEL_6;
  }
  *(_BYTE *)(*(_QWORD *)a6 + v9) = a3;
  ++*(_QWORD *)(a6 + 8);
  v12 = *v6;
  if ( a2 != *v6 )
  {
LABEL_7:
    v13 = 1;
    do
    {
      v14 = v13;
      v13 = *(_DWORD *)(a6 + 24);
      if ( v13 )
        break;
      if ( !v12 )
      {
        v36 = *(_QWORD *)(a6 + 32);
        ++*(_QWORD *)(a6 + 48);
        *(_DWORD *)(a6 + 24) = 1;
        v27 = v36 + 1;
        *(_QWORD *)(a6 + 32) = v27;
        goto LABEL_47;
      }
      if ( !v14 && !*(_QWORD *)(a6 + 32) )
        sub_8E5790((unsigned __int8 *)", ", a6);
      if ( a5 )
      {
LABEL_15:
        while ( *v6 == 100 )
        {
          while ( 1 )
          {
            v15 = v6[1];
LABEL_17:
            if ( v15 == 105 )
              break;
            if ( v15 == 120 )
            {
              if ( !*(_QWORD *)(a6 + 32) )
              {
                v31 = *(_QWORD *)(a6 + 8);
                v32 = v31 + 1;
                if ( !*(_DWORD *)(a6 + 28) )
                {
                  v33 = *(_QWORD *)(a6 + 16);
                  if ( v33 > v32 )
                  {
                    *(_BYTE *)(*(_QWORD *)a6 + v31) = 91;
                    v32 = *(_QWORD *)(a6 + 8) + 1LL;
                  }
                  else
                  {
                    *(_DWORD *)(a6 + 28) = 1;
                    if ( v33 )
                    {
                      *(_BYTE *)(*(_QWORD *)a6 + v33 - 1) = 0;
                      v32 = *(_QWORD *)(a6 + 8) + 1LL;
                    }
                  }
                }
                *(_QWORD *)(a6 + 8) = v32;
              }
              v34 = sub_8E74B0(v6 + 2, a6);
              v6 = v34;
              if ( !*(_QWORD *)(a6 + 32) )
                goto LABEL_29;
              if ( *v34 != 100 )
                goto LABEL_8;
            }
            else
            {
              if ( v15 != 88 )
                goto LABEL_8;
              if ( !*(_QWORD *)(a6 + 32) )
              {
                v16 = *(_QWORD *)(a6 + 8);
                v17 = v16 + 1;
                if ( !*(_DWORD *)(a6 + 28) )
                {
                  v18 = *(_QWORD *)(a6 + 16);
                  if ( v18 > v17 )
                  {
                    *(_BYTE *)(*(_QWORD *)a6 + v16) = 91;
                    v17 = *(_QWORD *)(a6 + 8) + 1LL;
                  }
                  else
                  {
                    *(_DWORD *)(a6 + 28) = 1;
                    if ( v18 )
                    {
                      *(_BYTE *)(*(_QWORD *)a6 + v18 - 1) = 0;
                      v17 = *(_QWORD *)(a6 + 8) + 1LL;
                    }
                  }
                }
                *(_QWORD *)(a6 + 8) = v17;
              }
              v19 = sub_8E74B0(v6 + 2, a6);
              if ( !*(_QWORD *)(a6 + 32) )
                sub_8E5790(" ... ", a6);
              v6 = sub_8E74B0(v19, a6);
              if ( *(_QWORD *)(a6 + 32) )
                goto LABEL_15;
LABEL_29:
              sub_8E5790("]=", a6);
              if ( *v6 != 100 )
                goto LABEL_8;
            }
          }
          do
          {
            if ( !*(_QWORD *)(a6 + 32) )
            {
              v20 = *(_QWORD *)(a6 + 8);
              v21 = v20 + 1;
              if ( !*(_DWORD *)(a6 + 28) )
              {
                v22 = *(_QWORD *)(a6 + 16);
                if ( v22 > v21 )
                {
                  *(_BYTE *)(*(_QWORD *)a6 + v20) = 46;
                  v21 = *(_QWORD *)(a6 + 8) + 1LL;
                }
                else
                {
                  *(_DWORD *)(a6 + 28) = 1;
                  if ( v22 )
                  {
                    *(_BYTE *)(*(_QWORD *)a6 + v22 - 1) = 0;
                    v21 = *(_QWORD *)(a6 + 8) + 1LL;
                  }
                }
              }
              *(_QWORD *)(a6 + 8) = v21;
            }
            v23 = sub_8E72C0(v6 + 2, 0, a6);
            v6 = v23;
            if ( *v23 != 100 )
            {
              if ( *(_QWORD *)(a6 + 32) )
                goto LABEL_8;
              goto LABEL_39;
            }
            v15 = v23[1];
          }
          while ( v15 == 105 );
          if ( *(_QWORD *)(a6 + 32) )
            goto LABEL_17;
LABEL_39:
          v24 = *(_QWORD *)(a6 + 8);
          v25 = v24 + 1;
          if ( !*(_DWORD *)(a6 + 28) )
          {
            v26 = *(_QWORD *)(a6 + 16);
            if ( v26 > v25 )
            {
              *(_BYTE *)(*(_QWORD *)a6 + v24) = 61;
              ++*(_QWORD *)(a6 + 8);
              continue;
            }
            *(_DWORD *)(a6 + 28) = 1;
            if ( v26 )
            {
              *(_BYTE *)(*(_QWORD *)a6 + v26 - 1) = 0;
              v25 = *(_QWORD *)(a6 + 8) + 1LL;
            }
          }
          *(_QWORD *)(a6 + 8) = v25;
        }
      }
LABEL_8:
      v6 = sub_8E74B0(v6, a6);
      v12 = *v6;
    }
    while ( *v6 != a2 );
  }
LABEL_46:
  v27 = *(_QWORD *)(a6 + 32);
LABEL_47:
  if ( !v27 )
  {
    v28 = *(_QWORD *)(a6 + 8);
    v29 = v28 + 1;
    if ( !*(_DWORD *)(a6 + 28) )
    {
      v35 = *(_QWORD *)(a6 + 16);
      if ( v35 > v29 )
      {
        *(_BYTE *)(*(_QWORD *)a6 + v28) = a4;
        v29 = *(_QWORD *)(a6 + 8) + 1LL;
      }
      else
      {
        *(_DWORD *)(a6 + 28) = 1;
        if ( v35 )
        {
          *(_BYTE *)(*(_QWORD *)a6 + v35 - 1) = 0;
          v29 = *(_QWORD *)(a6 + 8) + 1LL;
        }
      }
    }
    *(_QWORD *)(a6 + 8) = v29;
  }
  return v6;
}
