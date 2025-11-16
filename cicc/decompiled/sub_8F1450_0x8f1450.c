// Function: sub_8F1450
// Address: 0x8f1450
//
__int64 __fastcall sub_8F1450(unsigned int *a1, unsigned __int8 *a2, char *a3)
{
  unsigned __int8 *v5; // rbx
  unsigned __int8 v6; // r15
  int v7; // r12d
  int v8; // edx
  unsigned __int8 *v9; // rsi
  __int64 result; // rax
  signed __int64 v11; // r9
  char j; // di
  int v13; // r8d
  int v14; // ecx
  char *v15; // rax
  int v16; // ecx
  int v17; // edx
  char *v18; // rdx
  int v19; // esi
  int v20; // r8d
  bool v21; // bl
  int v22; // eax
  char *v23; // r15
  bool v24; // bl
  __int64 v25; // r14
  char i; // r8
  unsigned __int8 v27; // cl
  int v28; // r9d
  __int64 v29; // rdi
  char *v30; // rdi
  char v31; // si
  char *v32; // rsi
  char *v33; // r9
  char v34; // si
  unsigned int v35; // ecx
  int v36; // esi
  __int64 v37; // rcx
  unsigned int v38; // edi
  unsigned int v39; // ecx
  char *endptr; // [rsp+8h] [rbp-38h] BYREF

  v5 = a2;
  *((_QWORD *)a1 + 5) = a2;
  *((_QWORD *)a1 + 4) = a2;
  *(_QWORD *)a1 = 2;
  v6 = *a2;
  if ( !*a2 )
  {
LABEL_18:
    v7 = 0;
    goto LABEL_19;
  }
  while ( 1 )
  {
    v7 = isspace(v6);
    if ( !v7 )
      break;
    v6 = *++v5;
    if ( !v6 )
      goto LABEL_18;
  }
  if ( v6 == 45 )
  {
    a1[1] = 1;
    v8 = v5[1];
    v9 = v5 + 1;
    goto LABEL_7;
  }
  if ( v6 != 43 )
  {
    v8 = *v5;
    v9 = v5;
LABEL_7:
    result = v8 & 0xFFFFFFDF;
    if ( (v8 & 0xDF) == 0x4E )
      goto LABEL_8;
LABEL_48:
    if ( (_BYTE)result != 73 )
    {
      v5 = v9;
      if ( (_BYTE)v8 == 48 )
      {
        do
          ++v5;
        while ( *v5 == 48 );
        v7 = 1;
      }
LABEL_19:
      *((_QWORD *)a1 + 2) = v5;
      if ( (unsigned int)*v5 - 48 > 9 )
      {
        v15 = (char *)v5;
        v16 = 0;
      }
      else
      {
        v15 = (char *)v5;
        do
        {
          v16 = 1 - (_DWORD)v5 + (_DWORD)v15;
          v17 = (unsigned __int8)*++v15;
          if ( (unsigned int)(v17 - 48) > 9 )
          {
            v7 = 1;
            goto LABEL_24;
          }
        }
        while ( v15 != (char *)(v5 + 4990) );
        v7 = 1;
        v16 = 4990;
      }
LABEL_24:
      *((_QWORD *)a1 + 3) = v15;
      v18 = v15;
      if ( (unsigned int)(unsigned __int8)*v15 - 48 <= 9 )
      {
        do
          v19 = (unsigned __int8)*++v18;
        while ( (unsigned int)(v19 - 48) <= 9 );
      }
      v20 = 0;
      a1[2] = (_DWORD)v18 - (_DWORD)v5;
      if ( *v18 != 46 )
        goto LABEL_27;
      v30 = v18 + 1;
      if ( v5 == (unsigned __int8 *)v15 && v18[1] == 48 )
      {
        v36 = (_DWORD)v30 + (_DWORD)v18 - (_DWORD)v5;
        do
          a1[2] = v36 - (_DWORD)++v30;
        while ( *v30 == 48 );
        v7 = 1;
      }
      *((_QWORD *)a1 + 4) = v30;
      if ( (unsigned __int8)(*v30 - 48) > 9u || v16 > 4989 )
      {
        *((_QWORD *)a1 + 5) = v30;
        v33 = v30;
        v18 = v30;
      }
      else
      {
        v18 = v30;
        do
        {
          v31 = *++v18;
          ++v16;
        }
        while ( (unsigned __int8)(v31 - 48) <= 9u && v16 <= 4989 );
        *((_QWORD *)a1 + 5) = v18;
        if ( v30 != v18 )
        {
          v32 = v18;
          while ( *(v32 - 1) == 48 )
          {
            --v32;
            --v16;
            *((_QWORD *)a1 + 5) = v32;
            if ( v32 == v30 )
              goto LABEL_88;
          }
          v33 = (char *)*((_QWORD *)a1 + 5);
          v7 = 1;
          v20 = (_DWORD)v33 - (_DWORD)v30;
          if ( (unsigned __int8)(*v18 - 48) > 9u )
          {
LABEL_81:
            if ( v30 != v33 )
              goto LABEL_28;
            goto LABEL_27;
          }
          do
LABEL_80:
            v34 = *++v18;
          while ( (unsigned __int8)(v34 - 48) <= 9u );
          goto LABEL_81;
        }
LABEL_88:
        v33 = v30;
        v7 = 1;
      }
      v20 = 0;
      if ( (unsigned __int8)(*v18 - 48) > 9u )
      {
LABEL_27:
        if ( v5 != (unsigned __int8 *)v15 )
        {
          while ( *(v15 - 1) == 48 )
          {
            --v15;
            --v16;
            *((_QWORD *)a1 + 3) = v15;
            if ( v15 == (char *)v5 )
              goto LABEL_28;
          }
          v15 = (char *)*((_QWORD *)a1 + 3);
        }
LABEL_28:
        result = v15 - (char *)v5;
        a1[12] = result + v20;
        if ( !v16 )
          *a1 = 6;
        if ( v7 )
        {
          v21 = v16 == 0;
          if ( (*v18 & 0xDF) != 0x45 )
          {
            result = *a1;
            v24 = (_DWORD)result == 2 && v21;
            goto LABEL_53;
          }
          v22 = (unsigned __int8)v18[1];
          if ( (_BYTE)v22 == 45 )
          {
            v22 = (unsigned __int8)v18[2];
            v23 = v18 + 2;
          }
          else
          {
            if ( (_BYTE)v22 == 43 )
            {
              v22 = (unsigned __int8)v18[2];
              v23 = v18 + 2;
            }
            else
            {
              v23 = v18 + 1;
            }
            v7 = 0;
          }
          result = (unsigned int)(v22 - 48);
          if ( (unsigned int)result > 9 )
          {
            *a1 = 0;
LABEL_38:
            v18 = v23;
            goto LABEL_43;
          }
          result = strtol(v23, &endptr, 10);
          v18 = endptr;
          v37 = result;
          if ( endptr != v23 )
          {
            result = *a1;
            if ( (_DWORD)result == 6 )
              goto LABEL_43;
            if ( v37 > 2147483646 )
            {
              v23 = endptr;
              result = v7 == 0 ? 5 : 7;
              *a1 = result;
              goto LABEL_38;
            }
            v24 = (_DWORD)result == 2 && v21;
            v38 = a1[2] - v37;
            v39 = a1[2] + v37;
            if ( v7 )
              v39 = v38;
            a1[2] = v39;
LABEL_53:
            if ( v24 )
            {
              *a1 = 6;
            }
            else if ( (_DWORD)result == 2 )
            {
              v35 = a1[12];
              result = a1[2];
              if ( (int)(v35 + result) <= 5000 )
              {
                if ( (int)(v35 - result) > 5000 )
                {
                  result = (unsigned int)(result + 5000);
                  a1[12] = result;
                }
              }
              else
              {
                a1[12] = 5000 - result;
              }
            }
LABEL_43:
            if ( a3 == v18 )
              return result;
            goto LABEL_44;
          }
        }
        *a1 = 0;
        goto LABEL_43;
      }
      goto LABEL_80;
    }
    v25 = a3 - (char *)v9;
    if ( v25 <= 0 )
    {
      v29 = 0;
    }
    else
    {
      result = 0;
      for ( i = 105; ; i = aInfinity_1[result] )
      {
        v27 = v9[result];
        v28 = result;
        LODWORD(v29) = result;
        if ( v27 != i && v27 != aInfinity_0[result] )
          break;
        ++result;
        LODWORD(v29) = v28 + 1;
        if ( result == v25 )
          break;
      }
      if ( (_DWORD)v29 == 3 )
      {
        v29 = 3;
      }
      else
      {
        v29 = (int)v29;
        result = (__int64)"INFINITY";
        if ( aInfinity_0[(int)v29] )
          goto LABEL_44;
      }
    }
    if ( v25 == v29 )
    {
      *a1 = 4;
      return result;
    }
    goto LABEL_44;
  }
  LOBYTE(v8) = v5[1];
  v9 = v5 + 1;
  result = (unsigned __int8)v8 & 0xDF;
  if ( (v8 & 0xDF) != 0x4E )
    goto LABEL_48;
LABEL_8:
  v11 = a3 - (char *)v9;
  if ( a3 - (char *)v9 <= 0 )
  {
    if ( a3 == (char *)v9 )
    {
LABEL_17:
      *a1 = 3;
      return result;
    }
  }
  else
  {
    result = 0;
    for ( j = 110; ; j = aNvptxNan[result + 6] )
    {
      v13 = result;
      v14 = result;
      if ( (_BYTE)v8 != j && aNan_0[result] != (_BYTE)v8 )
        break;
      ++result;
      v14 = v13 + 1;
      if ( v11 == result )
        break;
      LOBYTE(v8) = v9[result];
    }
    if ( v14 == 3 )
    {
      if ( v11 == 3 )
        goto LABEL_17;
      if ( v9[3] == 40 )
      {
        result = (__int64)strchr((const char *)v9 + 4, 41);
        if ( a3 - 1 == (char *)result )
          goto LABEL_17;
      }
    }
    else
    {
      result = (__int64)"NAN";
      if ( !aNan_0[v14] && v11 == v14 )
        goto LABEL_17;
    }
  }
LABEL_44:
  *a1 = 0;
  return result;
}
