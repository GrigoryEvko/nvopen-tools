// Function: sub_228CF00
// Address: 0x228cf00
//
__int64 __fastcall sub_228CF00(__int64 a1, __int64 a2)
{
  char v3; // r13
  _DWORD *v4; // rdx
  unsigned int v5; // eax
  _WORD *v6; // rdx
  unsigned int v7; // r14d
  unsigned int i; // ebx
  char v9; // al
  __int64 v10; // rdi
  int v11; // eax
  char v12; // cl
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
  _QWORD *v24; // rdx
  char v25; // [rsp+Ch] [rbp-34h]
  char v26; // [rsp+Ch] [rbp-34h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1) )
  {
    v24 = *(_QWORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v24 <= 7u )
    {
      sub_CB6200(a2, "confused", 8u);
      v18 = *(_WORD **)(a2 + 32);
    }
    else
    {
      *v24 = 0x64657375666E6F63LL;
      v18 = (_WORD *)(*(_QWORD *)(a2 + 32) + 8LL);
      *(_QWORD *)(a2 + 32) = v18;
    }
    goto LABEL_42;
  }
  v3 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL))(a1) )
  {
    sub_904010(a2, "consistent ");
    if ( (unsigned __int8)sub_228CC90(a1) )
    {
LABEL_4:
      v4 = *(_DWORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 3u )
      {
        sub_CB6200(a2, (unsigned __int8 *)"flow", 4u);
      }
      else
      {
        *v4 = 2003790950;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      goto LABEL_6;
    }
  }
  else if ( (unsigned __int8)sub_228CC90(a1) )
  {
    goto LABEL_4;
  }
  if ( (unsigned __int8)sub_228CC50(a1) )
  {
    sub_904010(a2, "output");
  }
  else if ( (unsigned __int8)sub_228CCD0(a1) )
  {
    sub_904010(a2, "anti");
  }
  else if ( (unsigned __int8)sub_228CC10(a1) )
  {
    sub_904010(a2, "input");
  }
LABEL_6:
  v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 40LL))(a1);
  v6 = *(_WORD **)(a2 + 32);
  v7 = v5;
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 <= 1u )
  {
    sub_CB6200(a2, (unsigned __int8 *)" [", 2u);
  }
  else
  {
    *v6 = 23328;
    *(_QWORD *)(a2 + 32) += 2LL;
  }
  if ( v7 )
  {
    for ( i = 1; i <= v7; ++i )
    {
      while ( 1 )
      {
        v9 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 96LL))(a1, i);
        if ( v9 )
          v3 = v9;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 80LL))(a1, i) )
        {
          v14 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v14 >= *(_QWORD *)(a2 + 24) )
          {
            sub_CB5D20(a2, 112);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v14 + 1;
            *v14 = 112;
          }
        }
        v10 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 56LL))(a1, i);
        if ( v10 )
        {
          sub_D955C0(v10, a2);
        }
        else if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 104LL))(a1, i) )
        {
          v20 = *(_BYTE **)(a2 + 32);
          if ( *(_BYTE **)(a2 + 24) == v20 )
          {
            sub_CB6200(a2, (unsigned __int8 *)"S", 1u);
          }
          else
          {
            *v20 = 83;
            ++*(_QWORD *)(a2 + 32);
          }
        }
        else
        {
          v11 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 48LL))(a1, i);
          v12 = v11;
          if ( v11 == 7 )
          {
            v23 = *(_BYTE **)(a2 + 32);
            if ( *(_BYTE **)(a2 + 24) == v23 )
            {
              sub_CB6200(a2, (unsigned __int8 *)"*", 1u);
            }
            else
            {
              *v23 = 42;
              ++*(_QWORD *)(a2 + 32);
            }
          }
          else
          {
            if ( (v11 & 1) != 0 )
            {
              v22 = *(_BYTE **)(a2 + 32);
              if ( *(_BYTE **)(a2 + 24) == v22 )
              {
                v25 = v12;
                sub_CB6200(a2, "<", 1u);
                v12 = v25;
              }
              else
              {
                *v22 = 60;
                ++*(_QWORD *)(a2 + 32);
              }
            }
            if ( (v12 & 2) != 0 )
            {
              v21 = *(_BYTE **)(a2 + 32);
              if ( *(_BYTE **)(a2 + 24) == v21 )
              {
                v26 = v12;
                sub_CB6200(a2, (unsigned __int8 *)"=", 1u);
                v12 = v26;
              }
              else
              {
                *v21 = 61;
                ++*(_QWORD *)(a2 + 32);
              }
            }
            if ( (v12 & 4) != 0 )
            {
              v13 = *(_BYTE **)(a2 + 32);
              if ( *(_BYTE **)(a2 + 24) == v13 )
              {
                sub_CB6200(a2, (unsigned __int8 *)">", 1u);
              }
              else
              {
                *v13 = 62;
                ++*(_QWORD *)(a2 + 32);
              }
            }
          }
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 88LL))(a1, i) )
          break;
LABEL_12:
        if ( v7 > i )
          goto LABEL_34;
LABEL_13:
        if ( v7 < ++i )
          goto LABEL_36;
      }
      v15 = *(_BYTE **)(a2 + 32);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(a2 + 24) )
      {
        sub_CB5D20(a2, 112);
        goto LABEL_12;
      }
      *(_QWORD *)(a2 + 32) = v15 + 1;
      *v15 = 112;
      if ( v7 <= i )
        goto LABEL_13;
LABEL_34:
      v16 = *(_BYTE **)(a2 + 32);
      if ( *(_BYTE **)(a2 + 24) == v16 )
      {
        sub_CB6200(a2, (unsigned __int8 *)" ", 1u);
        goto LABEL_13;
      }
      *v16 = 32;
      ++*(_QWORD *)(a2 + 32);
    }
  }
LABEL_36:
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    sub_904010(a2, "|<");
    v17 = *(_BYTE **)(a2 + 32);
    if ( *(_BYTE **)(a2 + 24) != v17 )
      goto LABEL_38;
LABEL_60:
    sub_CB6200(a2, (unsigned __int8 *)"]", 1u);
    goto LABEL_39;
  }
  v17 = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == v17 )
    goto LABEL_60;
LABEL_38:
  *v17 = 93;
  ++*(_QWORD *)(a2 + 32);
LABEL_39:
  if ( v3 )
    sub_904010(a2, " splitable");
  v18 = *(_WORD **)(a2 + 32);
LABEL_42:
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v18 <= 1u )
    return sub_CB6200(a2, (unsigned __int8 *)"!\n", 2u);
  *v18 = 2593;
  *(_QWORD *)(a2 + 32) += 2LL;
  return 2593;
}
