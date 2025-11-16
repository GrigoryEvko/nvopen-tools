// Function: sub_7BBD70
// Address: 0x7bbd70
//
__int64 __fastcall sub_7BBD70(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // r14d
  int v15; // r13d
  __int64 result; // rax
  bool v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // [rsp+0h] [rbp-40h]
  unsigned int v23; // [rsp+Ch] [rbp-34h]

  v22 = a1;
  if ( (_DWORD)a2 )
  {
    sub_7AE360(a1);
    sub_7B8B50(a1, a2, v18, v19, v20, v21);
    sub_7AE360(a1);
  }
  else
  {
    sub_7B8B50(a1, a2, a3, a4, a5, a6);
  }
  sub_7B8B50(a1, a2, v6, v7, v8, v9);
  v14 = 0;
  v15 = 0;
  v23 = 0;
  result = word_4F06418[0];
  do
  {
    if ( (_WORD)result == 9 )
      return result;
    while ( 1 )
    {
      LOBYTE(v10) = (_WORD)result == 26;
      if ( (_WORD)result == 28 )
      {
        v15 = (v15 == 0) + v15 - 1;
        goto LABEL_10;
      }
      if ( (unsigned __int16)result > 0x1Cu )
      {
        if ( (_WORD)result == 73 )
        {
          ++v14;
          goto LABEL_10;
        }
        if ( (_WORD)result == 74 )
        {
          v14 = (v14 == 0) + v14 - 1;
          goto LABEL_10;
        }
        goto LABEL_26;
      }
      if ( (_WORD)result != 26 )
      {
        if ( (_WORD)result == 27 )
        {
          ++v15;
          goto LABEL_10;
        }
        if ( (_WORD)result == 25 )
        {
          if ( !(v14 | v15) )
          {
            ++v23;
            v14 = 0;
            v15 = 0;
            if ( !(_DWORD)a2 )
            {
LABEL_11:
              sub_7B8B50(a1, a2, v10, v11, v12, v13);
              goto LABEL_12;
            }
            goto LABEL_17;
          }
LABEL_10:
          if ( !(_DWORD)a2 )
            goto LABEL_11;
LABEL_17:
          v17 = 0;
LABEL_18:
          a1 = v22;
          sub_7AE360(v22);
          goto LABEL_19;
        }
LABEL_26:
        v11 = v23;
        v17 = v10 & (v23 == 0);
        goto LABEL_27;
      }
      if ( v14 | v15 )
      {
        v17 = v23 == 0;
        goto LABEL_27;
      }
      if ( !v23 )
        break;
      v17 = v23-- == 1;
      v14 = 0;
      v15 = 0;
LABEL_27:
      if ( (_DWORD)a2 )
        goto LABEL_18;
LABEL_19:
      sub_7B8B50(a1, a2, v10, v11, v12, v13);
      if ( v17 )
        goto LABEL_20;
LABEL_12:
      result = word_4F06418[0];
      if ( word_4F06418[0] == 9 )
        return result;
    }
    if ( (_DWORD)a2 )
    {
      v14 = 0;
      v15 = 0;
      v17 = 1;
      goto LABEL_18;
    }
    sub_7B8B50(a1, a2, v10, v11, v12, v13);
    v14 = 0;
    v15 = 0;
LABEL_20:
    result = word_4F06418[0];
  }
  while ( word_4F06418[0] != 26 );
  return result;
}
