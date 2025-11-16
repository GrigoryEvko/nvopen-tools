// Function: sub_681B50
// Address: 0x681b50
//
__int64 __fastcall sub_681B50(__int64 a1)
{
  char *v2; // r13
  __int64 result; // rax
  char *i; // rdx
  char v5; // cl
  char *v6; // r12
  int v7; // r14d
  char v8; // si
  int v9; // ecx
  const char *v10; // r12
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // rax
  const char *v14; // rdi
  __int64 v15; // r14
  __int64 v16; // r15
  _UNKNOWN **v17; // r13
  size_t v18; // r15
  int v19; // edi
  char *v20; // r12
  size_t v21; // rax
  char v22[80]; // [rsp+0h] [rbp-50h] BYREF

  v2 = sub_67C860(*(_DWORD *)(a1 + 176));
LABEL_2:
  while ( 1 )
  {
    result = (unsigned __int8)*v2;
    if ( !(_BYTE)result )
      break;
    for ( i = v2; ; ++i )
    {
      v5 = result;
      v6 = i + 1;
      result = (unsigned __int8)i[1];
      if ( v5 == 37 )
        break;
      if ( !(_BYTE)result )
        goto LABEL_20;
    }
    v7 = 1;
    if ( (_BYTE)result != 37 )
    {
      v6 = i;
      v7 = 0;
      --i;
    }
    if ( v2 <= i )
      goto LABEL_22;
LABEL_10:
    if ( !*v6 )
      return result;
    v2 = v6 + 1;
    if ( v7 )
      continue;
    v8 = v6[1];
    v9 = 1;
    v10 = v6 + 2;
    if ( v8 != 91 )
    {
      while ( 1 )
      {
        v11 = *v10;
        if ( (unsigned __int8)(*v10 - 48) <= 9u )
        {
          v9 = (char)(*v10 - 48);
        }
        else
        {
          if ( (unsigned __int8)((v11 & 0xDF) - 65) > 0x18u )
          {
            v2 = (char *)v10;
            v22[v7] = 0;
            sub_67FCF0((_QWORD *)a1, v8, v22, v9);
            goto LABEL_2;
          }
          v12 = v7++;
          v22[v12] = v11;
        }
        ++v10;
      }
    }
    v13 = sub_721BB0(v2, 93, i, 1);
    v14 = (const char *)off_4A44340;
    v15 = v13;
    v16 = v13 - (_QWORD)v2;
    v17 = &off_4A44340;
    v18 = v16 - 1;
    if ( off_4A44340 )
    {
      do
      {
        if ( !strncmp(v14, v10, v18) )
          break;
        v14 = (const char *)v17[3];
        v17 += 3;
      }
      while ( v14 );
    }
    v19 = *((_DWORD *)v17 + 5);
    if ( *(_DWORD *)v17[1] )
      v19 = *((_DWORD *)v17 + 4);
    v2 = (char *)(v15 + 1);
    v20 = sub_67C860(v19);
    v21 = strlen(v20);
    sub_8238B0(qword_4D039E8, v20, v21);
  }
  i = v2 - 1;
  v6 = v2;
LABEL_20:
  if ( v2 <= i )
  {
    v7 = 0;
LABEL_22:
    result = sub_8238B0(qword_4D039E8, v2, i - v2 + 1);
    goto LABEL_10;
  }
  return result;
}
