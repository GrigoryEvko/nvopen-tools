// Function: sub_8ECF80
// Address: 0x8ecf80
//
unsigned __int8 *__fastcall sub_8ECF80(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r13
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  unsigned __int8 *v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int8 v10; // dl
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rcx
  unsigned __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rcx
  __int64 v33; // rdx
  unsigned __int64 v34; // rax
  unsigned __int8 *v35; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1;
  v4 = a1[1];
  v35 = 0;
  if ( v4 == 83 )
  {
    ++*(_QWORD *)(a2 + 32);
    sub_8EC8F0((__int64)(a1 + 1), 0, 0, 0, 0, 0, &v35, a2);
    --*(_QWORD *)(a2 + 32);
    v4 = a1[1];
  }
  if ( v4 == 95 )
  {
    if ( a1[2] == 90 )
    {
      v2 = (unsigned __int8 *)sub_8E9250(a1 + 3, 0, a2);
      if ( *v2 == 69 )
        return ++v2;
    }
LABEL_46:
    if ( *(_DWORD *)(a2 + 24) )
      return v2;
    goto LABEL_47;
  }
  v5 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int8)(v4 - 100) <= 3u )
  {
    if ( !v5 )
    {
      v12 = *(_QWORD *)(a2 + 8);
      v13 = v12 + 1;
      if ( !*(_DWORD *)(a2 + 28) )
      {
        v14 = *(_QWORD *)(a2 + 16);
        if ( v14 > v13 )
        {
          *(_BYTE *)(*(_QWORD *)a2 + v12) = 40;
          v13 = *(_QWORD *)(a2 + 8) + 1LL;
        }
        else
        {
          *(_DWORD *)(a2 + 28) = 1;
          if ( v14 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v14 - 1) = 0;
            v13 = *(_QWORD *)(a2 + 8) + 1LL;
          }
        }
      }
      *(_QWORD *)(a2 + 8) = v13;
    }
    v2 = (unsigned __int8 *)sub_8E9FF0((__int64)(a1 + 1), 0, 0, 0, 1u, a2);
    sub_8EB260(a1 + 1, 0, 0, a2);
    if ( !*(_QWORD *)(a2 + 32) )
    {
      v15 = *(_QWORD *)(a2 + 8);
      v16 = v15 + 1;
      if ( !*(_DWORD *)(a2 + 28) )
      {
        v17 = *(_QWORD *)(a2 + 16);
        if ( v17 > v16 )
        {
          *(_BYTE *)(*(_QWORD *)a2 + v15) = 41;
          v16 = *(_QWORD *)(a2 + 8) + 1LL;
        }
        else
        {
          *(_DWORD *)(a2 + 28) = 1;
          if ( v17 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v17 - 1) = 0;
            v16 = *(_QWORD *)(a2 + 8) + 1LL;
          }
        }
      }
      *(_QWORD *)(a2 + 8) = v16;
    }
    if ( *(_DWORD *)(a2 + 24) )
      return v2;
    v2 = sub_8E59D0(v2, a2);
    if ( *(_DWORD *)(a2 + 24) )
      return v2;
    goto LABEL_43;
  }
  if ( v4 != 67 )
  {
    v6 = v35;
    if ( !v35 || *v35 != 67 )
    {
LABEL_8:
      if ( v4 == 68 && (a1[2] & 0xDF) == 0x4E && a1[3] == 69 )
      {
        *(_QWORD *)(a2 + 32) = v5 + 1;
        sub_8E9FF0((__int64)(a1 + 1), 0, 0, 0, 1u, a2);
        sub_8EB260(a1 + 1, 0, 0, a2);
        v25 = *(_QWORD *)(a2 + 32) - 1LL;
        *(_QWORD *)(a2 + 32) = v25;
        if ( a1[2] == 78 )
        {
          if ( !v25 )
            sub_8E5790("__nullptr", a2);
        }
        else if ( !v25 )
        {
          sub_8E5790((unsigned __int8 *)"nullptr", a2);
        }
        return a1 + 4;
      }
LABEL_9:
      if ( !v5 )
      {
        v7 = *(_QWORD *)(a2 + 8);
        v8 = v7 + 1;
        if ( !*(_DWORD *)(a2 + 28) )
        {
          v9 = *(_QWORD *)(a2 + 16);
          if ( v9 > v8 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v7) = 40;
            v8 = *(_QWORD *)(a2 + 8) + 1LL;
          }
          else
          {
            *(_DWORD *)(a2 + 28) = 1;
            if ( v9 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v9 - 1) = 0;
              v8 = *(_QWORD *)(a2 + 8) + 1LL;
            }
          }
        }
        *(_QWORD *)(a2 + 8) = v8;
      }
      qword_4F605D0 = 0;
      v2 = (unsigned __int8 *)sub_8E9FF0((__int64)(a1 + 1), 0, 0, 0, 1u, a2);
      sub_8EB260(a1 + 1, 0, 0, a2);
      if ( *(_QWORD *)(a2 + 32) )
      {
        v10 = *v2;
        if ( *v2 == 69 )
          return ++v2;
      }
      else
      {
        v22 = *(_QWORD *)(a2 + 8);
        v23 = v22 + 1;
        if ( !*(_DWORD *)(a2 + 28) )
        {
          v24 = *(_QWORD *)(a2 + 16);
          if ( v24 > v23 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v22) = 41;
            v23 = *(_QWORD *)(a2 + 8) + 1LL;
          }
          else
          {
            *(_DWORD *)(a2 + 28) = 1;
            if ( v24 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v24 - 1) = 0;
              v23 = *(_QWORD *)(a2 + 8) + 1LL;
            }
          }
        }
        *(_QWORD *)(a2 + 8) = v23;
        v10 = *v2;
        if ( *v2 == 69 )
        {
          if ( *(_QWORD *)(a2 + 32) )
            return ++v2;
          sub_8E5790((unsigned __int8 *)"\"...\"", a2);
LABEL_26:
          if ( *v2 == 69 )
            return ++v2;
          goto LABEL_46;
        }
      }
      if ( v10 == 110 )
      {
        if ( !*(_QWORD *)(a2 + 32) )
        {
          v29 = *(_QWORD *)(a2 + 8);
          v30 = v29 + 1;
          if ( !*(_DWORD *)(a2 + 28) )
          {
            v31 = *(_QWORD *)(a2 + 16);
            if ( v31 > v30 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v29) = 45;
              v30 = *(_QWORD *)(a2 + 8) + 1LL;
            }
            else
            {
              *(_DWORD *)(a2 + 28) = 1;
              if ( v31 )
              {
                *(_BYTE *)(*(_QWORD *)a2 + v31 - 1) = 0;
                v30 = *(_QWORD *)(a2 + 8) + 1LL;
              }
            }
          }
          *(_QWORD *)(a2 + 8) = v30;
        }
        v10 = *++v2;
      }
      if ( (unsigned int)v10 - 48 <= 9 )
      {
        do
        {
          if ( !*(_QWORD *)(a2 + 32) )
          {
            v20 = *(_QWORD *)(a2 + 8);
            v18 = v20 + 1;
            if ( !*(_DWORD *)(a2 + 28) )
            {
              v21 = *(_QWORD *)(a2 + 16);
              if ( v21 <= v18 )
              {
                *(_DWORD *)(a2 + 28) = 1;
                if ( v21 )
                {
                  *(_BYTE *)(*(_QWORD *)a2 + v21 - 1) = 0;
                  v18 = *(_QWORD *)(a2 + 8) + 1LL;
                }
              }
              else
              {
                *(_BYTE *)(*(_QWORD *)a2 + v20) = v10;
                v18 = *(_QWORD *)(a2 + 8) + 1LL;
              }
            }
            *(_QWORD *)(a2 + 8) = v18;
          }
          v19 = *++v2;
          v10 = v19;
        }
        while ( (unsigned int)(v19 - 48) <= 9 );
      }
      else if ( !dword_4D0425C && !*(_DWORD *)(a2 + 24) )
      {
        ++*(_QWORD *)(a2 + 32);
        ++*(_QWORD *)(a2 + 48);
        *(_DWORD *)(a2 + 24) = 1;
      }
      if ( qword_4F605D0 )
      {
        if ( !*(_QWORD *)(a2 + 32) )
          sub_8E5790((unsigned __int8 *)qword_4F605D0, a2);
        qword_4F605D0 = 0;
      }
      goto LABEL_26;
    }
    goto LABEL_72;
  }
  if ( (unsigned __int8)(a1[2] - 100) > 3u )
  {
    v6 = v35;
    if ( !v35 || *v35 != 67 )
      goto LABEL_9;
LABEL_72:
    if ( (unsigned __int8)(v6[1] - 100) > 3u )
      goto LABEL_8;
  }
  if ( !v5 )
  {
    v26 = *(_QWORD *)(a2 + 8);
    v27 = v26 + 1;
    if ( !*(_DWORD *)(a2 + 28) )
    {
      v32 = *(_QWORD *)(a2 + 16);
      if ( v32 > v27 )
      {
        *(_BYTE *)(*(_QWORD *)a2 + v26) = 40;
        v27 = *(_QWORD *)(a2 + 8) + 1LL;
      }
      else
      {
        *(_DWORD *)(a2 + 28) = 1;
        if ( v32 )
        {
          *(_BYTE *)(*(_QWORD *)a2 + v32 - 1) = 0;
          v27 = *(_QWORD *)(a2 + 8) + 1LL;
        }
      }
    }
    *(_QWORD *)(a2 + 8) = v27;
  }
  v2 = (unsigned __int8 *)sub_8E9FF0((__int64)(a1 + 1), 0, 0, 0, 1u, a2);
  sub_8EB260(a1 + 1, 0, 0, a2);
  if ( !*(_QWORD *)(a2 + 32) )
    sub_8E5790(")(", a2);
  if ( *(_DWORD *)(a2 + 24) )
    return v2;
  v2 = sub_8E59D0(v2, a2);
  if ( *(_DWORD *)(a2 + 24) )
    return v2;
  v28 = *(_QWORD *)(a2 + 32);
  if ( *v2 != 95 )
  {
    ++*(_QWORD *)(a2 + 48);
    *(_DWORD *)(a2 + 24) = 1;
    *(_QWORD *)(a2 + 32) = v28 + 1;
    return v2;
  }
  if ( !v28 )
  {
    if ( !*(_DWORD *)(a2 + 28) )
    {
      v33 = *(_QWORD *)(a2 + 8);
      v34 = *(_QWORD *)(a2 + 16);
      if ( v34 > v33 + 1 )
      {
        *(_BYTE *)(*(_QWORD *)a2 + v33) = 43;
      }
      else
      {
        *(_DWORD *)(a2 + 28) = 1;
        if ( v34 )
          *(_BYTE *)(*(_QWORD *)a2 + v34 - 1) = 0;
      }
    }
    ++*(_QWORD *)(a2 + 8);
  }
  v2 = sub_8E59D0(v2 + 1, a2);
  if ( *(_DWORD *)(a2 + 24) )
    return v2;
  if ( !*(_QWORD *)(a2 + 32) )
  {
    sub_8E5790((unsigned __int8 *)"i)", a2);
    if ( *(_DWORD *)(a2 + 24) )
      return v2;
  }
LABEL_43:
  if ( *v2 == 69 )
    return ++v2;
LABEL_47:
  ++*(_QWORD *)(a2 + 32);
  ++*(_QWORD *)(a2 + 48);
  *(_DWORD *)(a2 + 24) = 1;
  return v2;
}
