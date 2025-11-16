// Function: sub_220F040
// Address: 0x220f040
//
_DWORD *__fastcall sub_220F040(char a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  _DWORD *v5; // rdi
  _DWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  _DWORD *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // rdi

  *(_QWORD *)(a2 + 8) = a3;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 24) = 0;
  *(_DWORD *)a2 = 0;
  if ( !a1 )
  {
    a3[3] = a2;
    if ( (_QWORD *)a4[3] == a3 )
    {
      v5 = (_DWORD *)a4[1];
      a4[3] = a2;
      result = v5;
LABEL_6:
      if ( (_DWORD *)a2 == v5 )
      {
LABEL_17:
        *result = 1;
        return result;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v7 = *(_QWORD *)(a2 + 8);
          if ( *(_DWORD *)v7 )
            goto LABEL_17;
          v8 = *(_QWORD *)(v7 + 8);
          v9 = *(_QWORD *)(v8 + 16);
          if ( v7 != v9 )
          {
            if ( !v9 || *(_DWORD *)v9 )
            {
              if ( *(_QWORD *)(v7 + 16) == a2 )
              {
                v16 = *(_QWORD *)(a2 + 24);
                v17 = *(_QWORD *)(v7 + 8);
                *(_QWORD *)(v7 + 16) = v16;
                if ( v16 )
                {
                  *(_QWORD *)(v16 + 8) = v7;
                  v17 = *(_QWORD *)(v7 + 8);
                }
                *(_QWORD *)(a2 + 8) = v17;
                if ( v7 == a4[1] )
                {
                  a4[1] = a2;
                }
                else if ( v7 == *(_QWORD *)(v17 + 24) )
                {
                  *(_QWORD *)(v17 + 24) = a2;
                }
                else
                {
                  *(_QWORD *)(v17 + 16) = a2;
                }
                v18 = a2;
                *(_QWORD *)(a2 + 24) = v7;
                *(_QWORD *)(v7 + 8) = a2;
                a2 = v7;
                v7 = v18;
              }
              *(_DWORD *)v7 = 1;
              v11 = *(_QWORD *)(v8 + 24);
              *(_DWORD *)v8 = 0;
              v12 = *(_QWORD *)(v11 + 16);
              *(_QWORD *)(v8 + 24) = v12;
              if ( v12 )
                *(_QWORD *)(v12 + 8) = v8;
              v13 = *(_QWORD *)(v8 + 8);
              *(_QWORD *)(v11 + 8) = v13;
              if ( v8 == a4[1] )
              {
                a4[1] = v11;
              }
              else if ( v8 == *(_QWORD *)(v13 + 16) )
              {
                *(_QWORD *)(v13 + 16) = v11;
              }
              else
              {
                *(_QWORD *)(v13 + 24) = v11;
              }
              *(_QWORD *)(v11 + 16) = v8;
              *(_QWORD *)(v8 + 8) = v11;
              v5 = (_DWORD *)a4[1];
            }
            else
            {
              *(_DWORD *)v7 = 1;
              a2 = v8;
              *(_DWORD *)v9 = 1;
              *(_DWORD *)v8 = 0;
            }
            goto LABEL_11;
          }
          v10 = *(_DWORD **)(v8 + 24);
          if ( !v10 || *v10 )
            break;
          a2 = *(_QWORD *)(v7 + 8);
          *(_DWORD *)v7 = 1;
          *v10 = 1;
          *(_DWORD *)v8 = 0;
          result = v5;
          if ( v5 == (_DWORD *)a2 )
            goto LABEL_17;
        }
        if ( *(_QWORD *)(v7 + 24) == a2 )
          break;
LABEL_28:
        *(_DWORD *)v7 = 1;
        v14 = *(_QWORD *)(v9 + 24);
        *(_DWORD *)v8 = 0;
        *(_QWORD *)(v8 + 16) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 8) = v8;
        v15 = *(_QWORD *)(v8 + 8);
        *(_QWORD *)(v9 + 8) = v15;
        if ( v8 == a4[1] )
        {
          a4[1] = v9;
        }
        else if ( v8 == *(_QWORD *)(v15 + 24) )
        {
          *(_QWORD *)(v15 + 24) = v9;
        }
        else
        {
          *(_QWORD *)(v15 + 16) = v9;
        }
        *(_QWORD *)(v9 + 24) = v8;
        *(_QWORD *)(v8 + 8) = v9;
        v5 = (_DWORD *)a4[1];
LABEL_11:
        result = v5;
        if ( v5 == (_DWORD *)a2 )
          goto LABEL_17;
      }
      v19 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(v7 + 24) = v19;
      if ( v19 )
      {
        *(_QWORD *)(v19 + 8) = v7;
        v20 = *(_QWORD *)(v7 + 8);
        *(_QWORD *)(a2 + 8) = v20;
        if ( v7 != a4[1] )
        {
          if ( v7 != *(_QWORD *)(v20 + 16) )
          {
            *(_QWORD *)(v20 + 24) = a2;
LABEL_48:
            v21 = a2;
            *(_QWORD *)(a2 + 16) = v7;
            v9 = *(_QWORD *)(v8 + 16);
            *(_QWORD *)(v7 + 8) = a2;
            a2 = v7;
            v7 = v21;
            goto LABEL_28;
          }
LABEL_52:
          *(_QWORD *)(v20 + 16) = a2;
          goto LABEL_48;
        }
      }
      else
      {
        *(_QWORD *)(a2 + 8) = v8;
        v20 = v8;
        if ( v7 != a4[1] )
          goto LABEL_52;
      }
      a4[1] = a2;
      goto LABEL_48;
    }
LABEL_5:
    v5 = (_DWORD *)a4[1];
    result = v5;
    goto LABEL_6;
  }
  a3[2] = a2;
  if ( a4 != a3 )
  {
    if ( (_QWORD *)a4[2] == a3 )
      a4[2] = a2;
    goto LABEL_5;
  }
  a4[1] = a2;
  a4[3] = a2;
  *(_DWORD *)a2 = 1;
  return (_DWORD *)a2;
}
