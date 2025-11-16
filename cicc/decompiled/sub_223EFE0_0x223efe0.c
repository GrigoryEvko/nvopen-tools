// Function: sub_223EFE0
// Address: 0x223efe0
//
__int64 __fastcall sub_223EFE0(char *src, __int64 a2, char *a3, int *a4)
{
  char *v4; // r12
  char v5; // al
  char *v6; // rbx
  char *v8; // rbp
  char v9; // al
  unsigned int v10; // eax
  unsigned __int64 *v11; // rdx
  int v12; // eax
  __int64 result; // rax
  unsigned int v14; // eax
  char **v15; // rdx
  char *v16; // rdx
  char v17; // al

  v4 = &src[a2 - 1];
  v5 = *a3;
  if ( !*a3 )
  {
    v8 = src;
    result = 0;
    goto LABEL_19;
  }
  if ( src >= v4 )
  {
    v8 = src;
LABEL_32:
    sub_223EE70(src, (__int64)v8);
  }
  v6 = a3;
  v8 = src;
  while ( 1 )
  {
    if ( v5 != 37 )
      goto LABEL_4;
    v9 = v6[1];
    if ( v9 == 115 )
    {
      v14 = *a4;
      if ( (unsigned int)*a4 > 0x2F )
      {
        v15 = (char **)*((_QWORD *)a4 + 1);
        *((_QWORD *)a4 + 1) = v15 + 1;
      }
      else
      {
        v15 = (char **)(*((_QWORD *)a4 + 2) + v14);
        *a4 = v14 + 8;
      }
      v16 = *v15;
      v17 = *v16;
      if ( *v16 )
      {
        if ( v4 <= v8 )
        {
          v4 = v8;
LABEL_30:
          sub_223EE70(src, (__int64)v4);
        }
        while ( 1 )
        {
          ++v16;
          *v8++ = v17;
          v17 = *v16;
          if ( !*v16 )
            break;
          if ( v4 == v8 )
            goto LABEL_30;
        }
      }
      v6 += 2;
      goto LABEL_6;
    }
    if ( v9 != 122 )
    {
      if ( v9 == 37 )
        ++v6;
      else
LABEL_4:
        v9 = *v6;
      *v8 = v9;
      ++v6;
      ++v8;
LABEL_6:
      v5 = *v6;
      if ( !*v6 )
        break;
      goto LABEL_7;
    }
    if ( v6[2] != 117 )
      goto LABEL_4;
    v10 = *a4;
    if ( (unsigned int)*a4 > 0x2F )
    {
      v11 = (unsigned __int64 *)*((_QWORD *)a4 + 1);
      *((_QWORD *)a4 + 1) = v11 + 1;
    }
    else
    {
      v11 = (unsigned __int64 *)(*((_QWORD *)a4 + 2) + v10);
      *a4 = v10 + 8;
    }
    v12 = sub_223EF40(v8, v4 - v8, *v11);
    if ( v12 <= 0 )
      goto LABEL_32;
    v6 += 3;
    v8 += v12;
    v5 = *v6;
    if ( !*v6 )
      break;
LABEL_7:
    if ( v8 >= v4 )
      goto LABEL_32;
  }
  result = (unsigned int)((_DWORD)v8 - (_DWORD)src);
LABEL_19:
  *v8 = 0;
  return result;
}
