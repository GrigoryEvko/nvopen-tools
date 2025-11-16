// Function: sub_3012780
// Address: 0x3012780
//
unsigned __int64 __fastcall sub_3012780(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r9
  unsigned __int64 result; // rax
  int v5; // r11d
  __int64 *v6; // rbx
  __int64 *v7; // rcx
  __int64 *v8; // r8
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rbx

  v2 = *a1;
  v3 = *a1 >> 2;
  result = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = v3 & 1;
  if ( (v3 & 1) != 0 )
  {
    v7 = *(__int64 **)result;
    v9 = 8LL * *(unsigned int *)(result + 8);
    v8 = &v7[(unsigned __int64)v9 / 8];
    v10 = v9 >> 3;
    result = v9 >> 5;
    if ( result )
    {
      while ( *v7 != a2 )
      {
        if ( v7[1] == a2 )
        {
          ++v7;
          goto LABEL_15;
        }
        if ( v7[2] == a2 )
        {
          v7 += 2;
          goto LABEL_15;
        }
        if ( v7[3] == a2 )
        {
          v7 += 3;
          goto LABEL_15;
        }
        v7 += 4;
        if ( !--result )
        {
          v10 = v8 - v7;
          goto LABEL_7;
        }
      }
      goto LABEL_15;
    }
LABEL_7:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 == 1 )
        {
          v6 = v7;
          v7 = v8;
          goto LABEL_3;
        }
        v6 = v8;
LABEL_33:
        v8 = v6;
        v7 = v6;
LABEL_4:
        if ( (v3 & 1) != 1 )
          return result;
        goto LABEL_25;
      }
      if ( *v7 == a2 )
        goto LABEL_15;
      ++v7;
    }
    if ( *v7 == a2 )
      goto LABEL_15;
    v6 = v7 + 1;
    v7 = v8;
    goto LABEL_3;
  }
  v6 = a1;
  v7 = a1 + 1;
  if ( !result )
    goto LABEL_33;
LABEL_3:
  v8 = v7;
  if ( *v6 != a2 )
    goto LABEL_4;
  v7 = v6;
LABEL_15:
  if ( v7 == v8 )
    goto LABEL_4;
  result = (unsigned __int64)(v7 + 1);
  if ( v7 + 1 != v8 )
  {
    do
    {
      if ( *(_QWORD *)result != a2 )
        *v7++ = *(_QWORD *)result;
      result += 8LL;
    }
    while ( (__int64 *)result != v8 );
    v2 = *a1;
    v3 = *a1 >> 2;
    v5 = v3 & 1;
  }
  if ( v5 == 1 )
  {
LABEL_25:
    if ( v2 )
    {
      if ( (v3 & 1) != 0 )
      {
        v11 = v2 & 0xFFFFFFFFFFFFFFF8LL;
        v12 = v11;
        if ( v11 )
        {
          result = *(_QWORD *)v11;
          v13 = *(_QWORD *)v11 + 8LL * *(unsigned int *)(v11 + 8);
          v14 = v13 - (_QWORD)v8;
          if ( v8 != (__int64 *)v13 )
          {
            v7 = (__int64 *)memmove(v7, v8, v13 - (_QWORD)v8);
            result = *(_QWORD *)v12;
          }
          *(_DWORD *)(v12 + 8) = (__int64)((__int64)v7 + v14 - result) >> 3;
        }
      }
    }
    return result;
  }
  if ( v7 != v8 && a1 == v7 )
    *a1 = 0;
  return result;
}
