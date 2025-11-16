// Function: sub_31A9620
// Address: 0x31a9620
//
__int64 __fastcall sub_31A9620(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  char v10; // al
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  unsigned __int8 v20; // [rsp+Fh] [rbp-41h]
  __int64 v21; // [rsp+14h] [rbp-3Ch]

  v6 = (__int64)a3;
  v7 = a2 + 48;
  v8 = a4;
  v9 = *(_QWORD *)(a2 + 56);
  if ( a2 + 48 != v9 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v9 )
          BUG();
        v10 = *(_BYTE *)(v9 - 24);
        v11 = v9 - 24;
        if ( v10 != 85 )
          goto LABEL_11;
        v12 = *(_QWORD *)(v9 - 56);
        if ( v12 )
          break;
LABEL_9:
        if ( (unsigned __int8)sub_31A92A0(v9 - 24, v21, 0) )
        {
          if ( !*(_BYTE *)(v8 + 28) )
            goto LABEL_27;
          v19 = *(_QWORD **)(v8 + 8);
          a4 = *(unsigned int *)(v8 + 20);
          a3 = &v19[a4];
          if ( v19 == a3 )
            goto LABEL_29;
          while ( v11 != *v19 )
          {
            if ( a3 == ++v19 )
              goto LABEL_29;
          }
          v9 = *(_QWORD *)(v9 + 8);
          if ( v7 == v9 )
            return 1;
        }
        else
        {
          v10 = *(_BYTE *)(v9 - 24);
LABEL_11:
          if ( v10 == 61 )
          {
            v13 = *(_QWORD *)(v9 - 56);
            if ( *(_BYTE *)(v6 + 28) )
            {
              v14 = *(_QWORD **)(v6 + 8);
              a3 = &v14[*(unsigned int *)(v6 + 20)];
              if ( v14 != a3 )
              {
                while ( v13 != *v14 )
                {
                  if ( a3 == ++v14 )
                    goto LABEL_46;
                }
                goto LABEL_17;
              }
LABEL_46:
              if ( !*(_BYTE *)(v8 + 28) )
                goto LABEL_27;
              v18 = *(_QWORD **)(v8 + 8);
              a4 = *(unsigned int *)(v8 + 20);
              a3 = &v18[a4];
              if ( v18 == a3 )
                goto LABEL_29;
              while ( v11 != *v18 )
              {
                if ( a3 == ++v18 )
                  goto LABEL_29;
              }
              v9 = *(_QWORD *)(v9 + 8);
              if ( v7 == v9 )
                return 1;
            }
            else
            {
              if ( !sub_C8CA60(v6, v13) )
                goto LABEL_46;
LABEL_17:
              v9 = *(_QWORD *)(v9 + 8);
              if ( v7 == v9 )
                return 1;
            }
          }
          else if ( v10 == 62 )
          {
            if ( *(_BYTE *)(v8 + 28) )
            {
              v16 = *(_QWORD **)(v8 + 8);
              a4 = *(unsigned int *)(v8 + 20);
              a3 = &v16[a4];
              if ( v16 == a3 )
                goto LABEL_29;
              while ( v11 != *v16 )
              {
                if ( a3 == ++v16 )
                  goto LABEL_29;
              }
              goto LABEL_17;
            }
LABEL_27:
            sub_C8CC70(v8, v9 - 24, (__int64)a3, a4, a5, a6);
            v9 = *(_QWORD *)(v9 + 8);
            if ( v7 == v9 )
              return 1;
          }
          else
          {
            v20 = sub_B46420(v9 - 24);
            if ( v20 )
              return 0;
            if ( (unsigned __int8)sub_B46490(v9 - 24) || (unsigned __int8)sub_B46790((unsigned __int8 *)(v9 - 24), 0) )
              return v20;
            v9 = *(_QWORD *)(v9 + 8);
            if ( v7 == v9 )
              return 1;
          }
        }
      }
      if ( !*(_BYTE *)v12 && *(_QWORD *)(v12 + 24) == *(_QWORD *)(v9 + 56) && *(_DWORD *)(v12 + 36) == 11 )
      {
        if ( !*(_BYTE *)(v8 + 28) )
          goto LABEL_27;
        v17 = *(_QWORD **)(v8 + 8);
        a4 = *(unsigned int *)(v8 + 20);
        a3 = &v17[a4];
        if ( v17 == a3 )
        {
LABEL_29:
          if ( (unsigned int)a4 >= *(_DWORD *)(v8 + 16) )
            goto LABEL_27;
          a4 = (unsigned int)(a4 + 1);
          *(_DWORD *)(v8 + 20) = a4;
          *a3 = v11;
          ++*(_QWORD *)v8;
          v9 = *(_QWORD *)(v9 + 8);
          if ( v7 == v9 )
            return 1;
        }
        else
        {
          while ( v11 != *v17 )
          {
            if ( a3 == ++v17 )
              goto LABEL_29;
          }
          v9 = *(_QWORD *)(v9 + 8);
          if ( v7 == v9 )
            return 1;
        }
      }
      else
      {
        if ( *(_BYTE *)v12 )
          goto LABEL_9;
        a4 = *(_QWORD *)(v9 + 56);
        if ( *(_QWORD *)(v12 + 24) != a4 || (*(_BYTE *)(v12 + 33) & 0x20) == 0 || *(_DWORD *)(v12 + 36) != 155 )
          goto LABEL_9;
        v9 = *(_QWORD *)(v9 + 8);
        if ( v7 == v9 )
          return 1;
      }
    }
  }
  return 1;
}
