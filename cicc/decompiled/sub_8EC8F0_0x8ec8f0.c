// Function: sub_8EC8F0
// Address: 0x8ec8f0
//
_BYTE *__fastcall sub_8EC8F0(
        __int64 a1,
        int a2,
        int a3,
        int a4,
        int a5,
        const char **a6,
        unsigned __int8 **a7,
        __int64 a8)
{
  unsigned __int8 v9; // r15
  const char *v10; // rcx
  char *v11; // r13
  _BYTE *v12; // r13
  unsigned __int64 v14; // rax
  int v15; // esi
  int v16; // r15d
  char *v17; // rax
  char *v18; // r11
  unsigned __int8 *v19; // r10
  __int64 v20; // rax
  unsigned int v21; // edx
  unsigned __int8 *v22; // r10
  unsigned __int64 v23; // rsi
  int v24; // ecx
  unsigned __int8 *v26; // [rsp+0h] [rbp-80h]
  char *v28; // [rsp+8h] [rbp-78h]
  const char *v31; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v32; // [rsp+18h] [rbp-68h]
  int v33; // [rsp+28h] [rbp-58h] BYREF
  int v34; // [rsp+2Ch] [rbp-54h] BYREF
  _QWORD v35[10]; // [rsp+30h] [rbp-50h] BYREF

  v9 = *(_BYTE *)(a1 + 1);
  if ( a6 )
    *a6 = 0;
  if ( a7 )
    *a7 = 0;
  if ( islower(v9) )
  {
    v10 = "3std";
    v11 = "::std";
    if ( v9 != 116 )
    {
      v10 = "9allocator";
      v11 = "::std::allocator";
      if ( v9 != 97 )
      {
        v10 = "12basic_string";
        v11 = "::std::basic_string";
        if ( v9 != 98 )
        {
          v11 = "::std::basic_string<char, std::char_traits<char>, std::allocator<char> >";
          if ( v9 != 115 )
          {
            v10 = "13basic_istream";
            v11 = "::std::basic_istream<char, std::char_traits<char> >";
            if ( v9 != 105 )
            {
              v10 = "13basic_ostream";
              v11 = "::std::basic_ostream<char, std::char_traits<char> >";
              if ( v9 != 111 )
              {
                v10 = byte_3F871B3;
                v11 = (char *)byte_3F871B3;
                if ( v9 == 100 )
                {
                  v10 = "14basic_iostream";
                  v11 = "::std::basic_iostream<char, std::char_traits<char> >";
                }
              }
            }
          }
        }
      }
    }
    if ( a2 != 2 )
    {
      v31 = v10;
      sub_8E6E80(a3, 1, a8);
      v10 = v31;
      if ( !*(_QWORD *)(a8 + 32) )
      {
        sub_8E5790((unsigned __int8 *)v11, a8);
        v10 = v31;
      }
    }
    v12 = (_BYTE *)(a1 + 2);
    if ( a6 )
      *a6 = v10;
    return v12;
  }
  v12 = (_BYTE *)(a1 + 1);
  v14 = 0;
  if ( v9 != 95 )
  {
    v15 = *(char *)(a1 + 1);
    v16 = 0;
    while ( 1 )
    {
      v16 *= 36;
      if ( !(_BYTE)v15 )
        break;
      v17 = strchr("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", v15);
      if ( !v17 )
        break;
      v15 = (char)*++v12;
      v16 += v17 - "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      if ( (_BYTE)v15 == 95 )
        goto LABEL_27;
    }
    if ( !*(_DWORD *)(a8 + 24) )
    {
      ++*(_QWORD *)(a8 + 32);
      ++*(_QWORD *)(a8 + 48);
      *(_DWORD *)(a8 + 24) = 1;
    }
LABEL_27:
    v14 = (unsigned int)(v16 + 1);
  }
  if ( qword_4F605B8 > v14 )
  {
    if ( *v12 == 95 )
    {
      ++v12;
    }
    else if ( !*(_DWORD *)(a8 + 24) )
    {
      ++*(_QWORD *)(a8 + 32);
      ++*(_QWORD *)(a8 + 48);
      *(_DWORD *)(a8 + 24) = 1;
    }
    v18 = (char *)qword_4F605C0 + 32 * v14;
    v19 = *(unsigned __int8 **)v18;
    if ( a7 )
      *a7 = v19;
    v20 = *(_QWORD *)(a8 + 48);
    *(_QWORD *)(a8 + 48) = v20 + 1;
    if ( a2 == 2 )
    {
      if ( *((_DWORD *)v18 + 2) != 3 )
        goto LABEL_43;
    }
    else
    {
      v21 = *((_DWORD *)v18 + 2);
      if ( v21 != 3 )
      {
        v28 = v18;
        if ( v21 > 3 )
        {
          if ( v21 != 4 )
          {
            if ( !*(_DWORD *)(a8 + 24) )
            {
              ++*(_QWORD *)(a8 + 32);
              ++v20;
              *(_DWORD *)(a8 + 24) = 1;
            }
            goto LABEL_43;
          }
          sub_8E5C30((__int64)v19, a8);
          goto LABEL_42;
        }
        v32 = v19;
        if ( !v21 )
        {
          sub_8E6E80(a3, 1, a8);
          sub_8EC360(v32, v35, a8);
          v20 = *(_QWORD *)(a8 + 48) - 1LL;
          goto LABEL_43;
        }
        sub_8E6E80(a3, 1, a8);
        v22 = v32;
        v23 = *((_QWORD *)v28 + 2);
        if ( v23 )
        {
          v22 = sub_8EC3E0((__int64)v32, v23, &v33, &v34, v35, (__int64)a6, a8);
          if ( *((_DWORD *)v28 + 2) != 2 )
            goto LABEL_42;
          if ( *((_QWORD *)v28 + 2) && !*(_QWORD *)(a8 + 32) )
            sub_8E5790((unsigned __int8 *)"::", a8);
        }
        else if ( *((_DWORD *)v28 + 2) != 2 )
        {
LABEL_42:
          v20 = *(_QWORD *)(a8 + 48) - 1LL;
LABEL_43:
          *(_QWORD *)(a8 + 48) = v20;
          return v12;
        }
        sub_8EBEA0(v22, &v33, a8);
        goto LABEL_42;
      }
      v24 = a5;
      v26 = v19;
      sub_8E9FF0((__int64)v19, a3, a4, v24, *((_DWORD *)v18 + 6), a8);
      v19 = v26;
      if ( a2 )
        goto LABEL_42;
    }
    sub_8EB260(v19, a3, a4, a8);
    v20 = *(_QWORD *)(a8 + 48) - 1LL;
    goto LABEL_43;
  }
  if ( !*(_DWORD *)(a8 + 24) )
  {
    ++*(_QWORD *)(a8 + 32);
    ++*(_QWORD *)(a8 + 48);
    *(_DWORD *)(a8 + 24) = 1;
  }
  return v12;
}
