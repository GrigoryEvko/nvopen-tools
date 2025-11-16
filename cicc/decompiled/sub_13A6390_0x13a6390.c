// Function: sub_13A6390
// Address: 0x13a6390
//
__int64 __fastcall sub_13A6390(__int64 a1, __int64 a2)
{
  char v3; // r13
  _DWORD *v4; // rdx
  unsigned int v5; // eax
  _WORD *v6; // rdx
  unsigned int v7; // r14d
  unsigned int i; // ebx
  char v9; // al
  __int64 v10; // rdi
  unsigned int v11; // eax
  __int64 v12; // rcx
  _BYTE *v13; // rax
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  _WORD *v18; // rdx
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  char v24; // [rsp+Ch] [rbp-34h]
  char v25; // [rsp+Ch] [rbp-34h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1) )
  {
    sub_1263B40(a2, "confused");
    goto LABEL_41;
  }
  v3 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL))(a1) )
  {
    sub_1263B40(a2, "consistent ");
    if ( (unsigned __int8)sub_13A6120(a1) )
    {
LABEL_4:
      v4 = *(_DWORD **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 3u )
      {
        sub_16E7EE0(a2, "flow", 4);
      }
      else
      {
        *v4 = 2003790950;
        *(_QWORD *)(a2 + 24) += 4LL;
      }
      goto LABEL_6;
    }
  }
  else if ( (unsigned __int8)sub_13A6120(a1) )
  {
    goto LABEL_4;
  }
  if ( (unsigned __int8)sub_13A60E0(a1) )
  {
    sub_1263B40(a2, "output");
  }
  else if ( (unsigned __int8)sub_13A6160(a1) )
  {
    sub_1263B40(a2, "anti");
  }
  else if ( (unsigned __int8)sub_13A60A0(a1) )
  {
    sub_1263B40(a2, "input");
  }
LABEL_6:
  v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 40LL))(a1);
  v6 = *(_WORD **)(a2 + 24);
  v7 = v5;
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v6 <= 1u )
  {
    sub_16E7EE0(a2, " [", 2);
  }
  else
  {
    *v6 = 23328;
    *(_QWORD *)(a2 + 24) += 2LL;
  }
  if ( v7 )
  {
    for ( i = 1; i <= v7; ++i )
    {
      while ( 1 )
      {
        v9 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 80LL))(a1, i);
        if ( v9 )
          v3 = v9;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 64LL))(a1, i) )
        {
          v14 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v14 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 112);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v14 + 1;
            *v14 = 112;
          }
        }
        v10 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 56LL))(a1, i);
        if ( v10 )
        {
          sub_1456620(v10, a2);
        }
        else if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 88LL))(a1, i) )
        {
          v20 = *(_BYTE **)(a2 + 24);
          if ( *(_BYTE **)(a2 + 16) == v20 )
          {
            sub_16E7EE0(a2, "S", 1);
          }
          else
          {
            *v20 = 83;
            ++*(_QWORD *)(a2 + 24);
          }
        }
        else
        {
          v11 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 48LL))(a1, i);
          v12 = v11;
          if ( v11 == 7 )
          {
            v23 = *(_BYTE **)(a2 + 24);
            if ( *(_BYTE **)(a2 + 16) == v23 )
            {
              sub_16E7EE0(a2, "*", 1, v12);
            }
            else
            {
              *v23 = 42;
              ++*(_QWORD *)(a2 + 24);
            }
          }
          else
          {
            if ( (v11 & 1) != 0 )
            {
              v22 = *(_BYTE **)(a2 + 24);
              if ( *(_BYTE **)(a2 + 16) == v22 )
              {
                v24 = v12;
                sub_16E7EE0(a2, "<", 1);
                LOBYTE(v12) = v24;
              }
              else
              {
                *v22 = 60;
                ++*(_QWORD *)(a2 + 24);
              }
            }
            if ( (v12 & 2) != 0 )
            {
              v21 = *(_BYTE **)(a2 + 24);
              if ( *(_BYTE **)(a2 + 16) == v21 )
              {
                v25 = v12;
                sub_16E7EE0(a2, "=", 1);
                LOBYTE(v12) = v25;
              }
              else
              {
                *v21 = 61;
                ++*(_QWORD *)(a2 + 24);
              }
            }
            if ( (v12 & 4) != 0 )
            {
              v13 = *(_BYTE **)(a2 + 24);
              if ( *(_BYTE **)(a2 + 16) == v13 )
              {
                sub_16E7EE0(a2, ">", 1);
              }
              else
              {
                *v13 = 62;
                ++*(_QWORD *)(a2 + 24);
              }
            }
          }
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 72LL))(a1, i) )
          break;
LABEL_12:
        if ( v7 > i )
          goto LABEL_34;
LABEL_13:
        if ( v7 < ++i )
          goto LABEL_36;
      }
      v15 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(a2 + 16) )
      {
        sub_16E7DE0(a2, 112);
        goto LABEL_12;
      }
      *(_QWORD *)(a2 + 24) = v15 + 1;
      *v15 = 112;
      if ( v7 <= i )
        goto LABEL_13;
LABEL_34:
      v16 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == v16 )
      {
        sub_16E7EE0(a2, " ", 1);
        goto LABEL_13;
      }
      *v16 = 32;
      ++*(_QWORD *)(a2 + 24);
    }
  }
LABEL_36:
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    sub_1263B40(a2, "|<");
    v17 = *(_BYTE **)(a2 + 24);
    if ( *(_BYTE **)(a2 + 16) != v17 )
      goto LABEL_38;
  }
  else
  {
    v17 = *(_BYTE **)(a2 + 24);
    if ( *(_BYTE **)(a2 + 16) != v17 )
    {
LABEL_38:
      *v17 = 93;
      ++*(_QWORD *)(a2 + 24);
      goto LABEL_39;
    }
  }
  sub_16E7EE0(a2, "]", 1);
LABEL_39:
  if ( v3 )
    sub_1263B40(a2, " splitable");
LABEL_41:
  v18 = *(_WORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v18 <= 1u )
    return sub_16E7EE0(a2, "!\n", 2);
  *v18 = 2593;
  *(_QWORD *)(a2 + 24) += 2LL;
  return 2593;
}
