// Function: sub_85E280
// Address: 0x85e280
//
__int64 __fastcall sub_85E280(char *a1, __int64 a2)
{
  char *v2; // r14
  char v3; // cl
  __int64 v4; // rbx
  char v5; // al
  __int64 result; // rax
  bool v7; // r15
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // rbx
  const char *v12; // r12
  char v13; // al
  _QWORD *v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rbx
  __int64 v17; // rax
  char *v18; // r13
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // r12
  _QWORD *v24; // [rsp+8h] [rbp-38h]

  v2 = a1;
  v3 = a1[80];
  v4 = qword_4F04C68[0] + 776LL * (int)a2;
  v5 = *(_BYTE *)(v4 + 4);
  if ( ((v5 - 15) & 0xFD) != 0 && v5 != 2 )
  {
    if ( v3 == 4 && (*(_BYTE *)(*(_QWORD *)(*((_QWORD *)a1 + 11) + 168LL) + 109LL) & 0x20) != 0 )
    {
      v23 = *((_QWORD *)a1 + 12);
      result = sub_6E26B0();
      if ( !(_DWORD)result )
      {
        result = *(_QWORD *)(v4 + 696) + 1LL;
        *(_QWORD *)(v4 + 696) = result;
        *(_QWORD *)(v23 + 168) = result;
      }
    }
    else
    {
      result = sub_879510(a1);
      if ( (_DWORD)result )
      {
        result = (unsigned __int8)a1[80];
        if ( (unsigned __int8)(result - 4) <= 1u )
        {
          result = *((_QWORD *)a1 + 11);
          if ( result )
          {
            if ( (*(_BYTE *)(result + 177) & 0x20) == 0 && !*(_QWORD *)(result + 8) )
            {
              result = *((_QWORD *)a1 + 12);
              if ( !*(_QWORD *)(result + 168) )
              {
                result = *(_QWORD *)(v4 + 688) + 1LL;
                *(_QWORD *)(v4 + 688) = result;
                *(_QWORD *)(*((_QWORD *)a1 + 12) + 168LL) = result;
              }
            }
          }
        }
        else if ( (_BYTE)result == 6 )
        {
          result = *((_QWORD *)a1 + 11);
          if ( !*(_QWORD *)(result + 8) )
          {
            result = *((_QWORD *)a1 + 12);
            if ( !*(_QWORD *)(result + 16) )
            {
              result = *(_QWORD *)(v4 + 688) + 1LL;
              *(_QWORD *)(v4 + 688) = result;
              *(_QWORD *)(*((_QWORD *)a1 + 12) + 16LL) = result;
            }
          }
        }
      }
    }
    return result;
  }
  v7 = 0;
  if ( v3 == 4 )
  {
    a2 = (unsigned int)a2;
    v7 = (unsigned int)sub_87A660(a1) != 0;
  }
  v8 = (_QWORD *)sub_85B7C0(a1, a2);
  v11 = (_QWORD *)*v8;
  v24 = v8;
  if ( !*v8 )
  {
LABEL_42:
    v16 = 1;
    goto LABEL_43;
  }
  while ( 1 )
  {
    v14 = (_QWORD *)v11[1];
    if ( (unsigned int)sub_879510(v2) )
    {
      v12 = (const char *)sub_85B780(v2);
      if ( !(unsigned int)sub_879510(v14) )
        goto LABEL_23;
    }
    else
    {
      v12 = *(const char **)(*(_QWORD *)v2 + 8LL);
      if ( !(unsigned int)sub_879510(v14) )
      {
LABEL_23:
        a2 = *(_QWORD *)(*v14 + 8LL);
        goto LABEL_16;
      }
    }
    a2 = sub_85B780(v14);
LABEL_16:
    if ( (const char *)a2 != v12 )
    {
      a1 = (char *)v12;
      if ( strcmp(v12, (const char *)a2) )
        goto LABEL_20;
    }
    a1 = (char *)v11[1];
    v13 = a1[80];
    if ( !v7 )
      break;
    if ( v13 == 4 && (unsigned int)sub_87A660(a1) )
    {
      v17 = v11[1];
      v18 = (char *)*((_QWORD *)v2 + 11);
      a1 = *(char **)(v17 + 88);
      if ( dword_4D0425C )
        goto LABEL_38;
      v19 = sub_72F130(a1);
      a1 = v18;
      v20 = v19;
      v21 = sub_72F130(v18);
      if ( v20 )
      {
        if ( v21 )
        {
          a2 = *(_QWORD *)(v21 + 152);
          a1 = *(char **)(v20 + 152);
          if ( (unsigned int)sub_8DE890(a1, a2, 0, 0) )
          {
            v17 = v11[1];
LABEL_38:
            v9 = *(unsigned __int8 *)(v17 + 80);
            if ( (_BYTE)v9 == 6 )
            {
              v15 = *(_QWORD *)(*(_QWORD *)(v17 + 96) + 16LL);
            }
            else
            {
              if ( (unsigned __int8)v9 <= 6u )
              {
                if ( (_BYTE)v9 == 3 )
                {
                  v15 = *(_QWORD *)(v17 + 96);
                  goto LABEL_31;
                }
                v9 = (unsigned int)(v9 - 4);
                if ( (unsigned __int8)v9 <= 1u )
                {
                  v15 = *(_QWORD *)(*(_QWORD *)(v17 + 96) + 168LL);
                  goto LABEL_31;
                }
LABEL_50:
                sub_721090();
              }
              if ( (_BYTE)v9 != 7 )
                goto LABEL_50;
              v15 = *(_QWORD *)(v17 + 104);
            }
LABEL_31:
            v16 = v15 + 1;
            goto LABEL_43;
          }
        }
      }
    }
LABEL_20:
    v11 = (_QWORD *)*v11;
    if ( !v11 )
      goto LABEL_42;
  }
  if ( v13 != 4 )
    goto LABEL_27;
  if ( (unsigned int)sub_87A660(a1) )
    goto LABEL_20;
  a1 = (char *)v11[1];
  v13 = a1[80];
LABEL_27:
  if ( v13 == 6 )
  {
    v15 = *(_QWORD *)(*((_QWORD *)a1 + 12) + 16LL);
    goto LABEL_31;
  }
  if ( (unsigned __int8)v13 > 6u )
  {
    if ( v13 == 7 )
    {
      v15 = *((_QWORD *)a1 + 13);
      goto LABEL_31;
    }
    goto LABEL_50;
  }
  if ( v13 == 3 )
  {
    v15 = *((_QWORD *)a1 + 12);
    goto LABEL_31;
  }
  if ( (unsigned __int8)(v13 - 4) > 1u )
    goto LABEL_50;
  v16 = *(_QWORD *)(*((_QWORD *)a1 + 12) + 168LL) + 1LL;
LABEL_43:
  v22 = (_QWORD *)sub_878440(a1, a2, v9, v10);
  *v22 = *v24;
  *v24 = v22;
  v22[1] = v2;
  result = (unsigned __int8)v2[80];
  if ( (_BYTE)result == 6 )
  {
    result = *((_QWORD *)v2 + 12);
    *(_QWORD *)(result + 16) = v16;
  }
  else if ( (unsigned __int8)result > 6u )
  {
    if ( (_BYTE)result != 7 )
      goto LABEL_50;
    *((_QWORD *)v2 + 13) = v16;
  }
  else if ( (_BYTE)result == 3 )
  {
    *((_QWORD *)v2 + 12) = v16;
  }
  else
  {
    if ( (unsigned __int8)(result - 4) > 1u )
      goto LABEL_50;
    result = *((_QWORD *)v2 + 12);
    *(_QWORD *)(result + 168) = v16;
  }
  return result;
}
