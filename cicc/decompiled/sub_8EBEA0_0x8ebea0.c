// Function: sub_8EBEA0
// Address: 0x8ebea0
//
unsigned __int8 *__fastcall sub_8EBEA0(unsigned __int8 *a1, _DWORD *a2, __int64 a3)
{
  _DWORD *v3; // r9
  int v5; // edx
  __int64 v6; // rax
  unsigned __int8 *v7; // rax
  unsigned __int8 *v8; // r13
  unsigned __int8 v10; // al
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rcx
  unsigned __int8 *v18; // rax
  __int64 v19; // rax
  int v20; // [rsp+4h] [rbp-6Ch] BYREF
  __int64 v21; // [rsp+8h] [rbp-68h] BYREF
  char s[8]; // [rsp+10h] [rbp-60h] BYREF

  v3 = a2;
  if ( a2 )
    *a2 = 0;
  v5 = *a1;
  if ( (unsigned int)(v5 - 48) <= 9 )
  {
    v8 = sub_8E72C0(a1, 0, a3);
    if ( *v8 != 66 )
      return v8;
    return sub_8E5930(v8, a3);
  }
  if ( (_BYTE)v5 == 85 )
  {
    v10 = a1[1];
    if ( v10 == 116 )
    {
      v8 = sub_8E5D20(a1 + 2, &v21, a3);
      if ( !*(_DWORD *)(a3 + 24) )
      {
        if ( !*(_QWORD *)(a3 + 32) )
          sub_8E5790("[unnamed type (instance ", a3);
        sprintf(s, "%lu", v21);
        if ( !*(_QWORD *)(a3 + 32) )
        {
          sub_8E5790((unsigned __int8 *)s, a3);
          if ( !*(_QWORD *)(a3 + 32) )
            sub_8E5790(")]", a3);
        }
      }
      goto LABEL_12;
    }
    if ( v10 == 108 )
    {
      if ( !*(_QWORD *)(a3 + 32) )
        sub_8E5790("[lambda", a3);
      v18 = sub_8EBA20(a1 + 2, 1, 2, a3);
      v8 = v18;
      if ( *v18 == 69 )
      {
        v8 = sub_8E5D20(v18 + 1, &v21, a3);
        if ( !*(_DWORD *)(a3 + 24) )
        {
          if ( !*(_QWORD *)(a3 + 32) )
            sub_8E5790((unsigned __int8 *)" (instance ", a3);
          sprintf(s, "%lu", v21);
          if ( !*(_QWORD *)(a3 + 32) )
            sub_8E5790((unsigned __int8 *)s, a3);
          if ( *(_QWORD *)(a3 + 32) )
            goto LABEL_12;
          sub_8E5790((unsigned __int8 *)")", a3);
        }
        v19 = *(_QWORD *)(a3 + 32);
      }
      else
      {
        v19 = *(_QWORD *)(a3 + 32);
        if ( !*(_DWORD *)(a3 + 24) )
        {
          ++v19;
          ++*(_QWORD *)(a3 + 48);
          *(_DWORD *)(a3 + 24) = 1;
          *(_QWORD *)(a3 + 32) = v19;
        }
      }
      if ( !v19 )
        sub_8E5790((unsigned __int8 *)"]", a3);
      goto LABEL_12;
    }
  }
  else
  {
    v6 = *(_QWORD *)(a3 + 32);
    if ( (_BYTE)v5 != 68 )
    {
      if ( v6 )
      {
LABEL_7:
        if ( (_BYTE)v5 == 99 && a1[1] == 118 )
        {
          if ( v3 )
            *v3 = 1;
          v8 = (unsigned __int8 *)sub_8E9FF0((__int64)(a1 + 2), 0, 0, 0, *(_DWORD *)(a3 + 60), a3);
          sub_8EB260(a1 + 2, 0, 0, a3);
          *(_DWORD *)(a3 + 56) = 1;
          goto LABEL_12;
        }
        goto LABEL_9;
      }
LABEL_17:
      sub_8E5790("operator ", a3);
      LOBYTE(v5) = *a1;
      goto LABEL_7;
    }
    if ( a1[1] == 67 )
    {
      v8 = a1 + 2;
      if ( v6 )
      {
        v11 = a1[2];
        if ( v11 != 69 )
          goto LABEL_28;
        goto LABEL_55;
      }
      sub_8E5790("[structured binding for ", a3);
      v11 = a1[2];
      if ( v11 == 69 )
      {
LABEL_49:
        if ( !*(_QWORD *)(a3 + 32) )
        {
          v15 = *(_QWORD *)(a3 + 8);
          v16 = v15 + 1;
          if ( !*(_DWORD *)(a3 + 28) )
          {
            v17 = *(_QWORD *)(a3 + 16);
            if ( v17 > v16 )
            {
              *(_BYTE *)(*(_QWORD *)a3 + v15) = 93;
              v16 = *(_QWORD *)(a3 + 8) + 1LL;
            }
            else
            {
              *(_DWORD *)(a3 + 28) = 1;
              if ( v17 )
              {
                *(_BYTE *)(*(_QWORD *)a3 + v17 - 1) = 0;
                v16 = *(_QWORD *)(a3 + 8) + 1LL;
              }
            }
          }
          *(_QWORD *)(a3 + 8) = v16;
        }
LABEL_55:
        ++v8;
        goto LABEL_12;
      }
      while ( 1 )
      {
        while ( 1 )
        {
LABEL_28:
          if ( !v11 )
            goto LABEL_40;
          v8 = sub_8E72C0(v8, 0, a3);
          v11 = *v8;
          if ( *v8 == 69 )
            goto LABEL_49;
          if ( v11 )
            break;
LABEL_37:
          if ( v11 == 69 )
            goto LABEL_49;
        }
        if ( !*(_QWORD *)(a3 + 32) )
        {
          v12 = *(_QWORD *)(a3 + 8);
          v13 = v12 + 1;
          if ( !*(_DWORD *)(a3 + 28) )
          {
            v14 = *(_QWORD *)(a3 + 16);
            if ( v14 > v13 )
            {
              *(_BYTE *)(*(_QWORD *)a3 + v12) = 44;
              v13 = *(_QWORD *)(a3 + 8) + 1LL;
            }
            else
            {
              *(_DWORD *)(a3 + 28) = 1;
              if ( v14 )
              {
                *(_BYTE *)(*(_QWORD *)a3 + v14 - 1) = 0;
                v13 = *(_QWORD *)(a3 + 8) + 1LL;
              }
            }
          }
          *(_QWORD *)(a3 + 8) = v13;
          v11 = *v8;
          goto LABEL_37;
        }
      }
    }
  }
  if ( !*(_QWORD *)(a3 + 32) )
    goto LABEL_17;
LABEL_9:
  v7 = (unsigned __int8 *)sub_8E6070(a1, &v20, (int *)&v21, s, a3);
  if ( v7 )
  {
    if ( !*(_QWORD *)(a3 + 32) )
    {
      sub_8E5790(v7, a3);
      if ( !*(_QWORD *)(a3 + 32) )
        sub_8E5790(*(unsigned __int8 **)s, a3);
    }
    v8 = &a1[(int)v21];
  }
  else
  {
    v8 = a1;
LABEL_40:
    if ( !*(_DWORD *)(a3 + 24) )
    {
      ++*(_QWORD *)(a3 + 32);
      ++*(_QWORD *)(a3 + 48);
      *(_DWORD *)(a3 + 24) = 1;
    }
  }
LABEL_12:
  if ( *v8 != 66 )
    return v8;
  return sub_8E5930(v8, a3);
}
