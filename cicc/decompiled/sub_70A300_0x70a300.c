// Function: sub_70A300
// Address: 0x70a300
//
__int64 __fastcall sub_70A300(int *a1, _QWORD *a2, int a3, int a4, int a5, _DWORD *a6)
{
  int v8; // ebx
  unsigned int *v9; // r13
  char v10; // cl
  unsigned int v11; // r15d
  __int64 result; // rax
  unsigned int v13; // ecx
  __int64 v14; // rcx
  unsigned int v15; // edx
  _DWORD *v16; // r9
  unsigned int v17; // eax
  int v18; // ebx
  __int64 v19; // rax
  int v21; // esi
  int v22; // edx
  unsigned int v24; // [rsp+10h] [rbp-40h]
  int v26; // [rsp+18h] [rbp-38h]

  v8 = a3 / 32;
  v9 = (unsigned int *)&a1[a3 / 32];
  v10 = a3 % 32;
  v11 = 0xFFFFFFFF >> v10;
  result = 0x80000000 >> v10;
  v13 = (0xFFFFFFFF >> v10) & *v9;
  if ( (unsigned int)result > v13 )
    return result;
  v24 = a1[4];
  if ( (unsigned int)result < v13 || a4 )
    goto LABEL_8;
  LODWORD(v14) = v8 + 1;
  if ( v8 + 1 <= 3 )
  {
    v14 = (int)v14;
    while ( !a1[v14] )
    {
      if ( (int)++v14 > 3 )
        goto LABEL_18;
    }
    goto LABEL_8;
  }
LABEL_18:
  if ( v24 )
    goto LABEL_8;
  v21 = a3 + 30;
  v22 = a3 - 1;
  if ( v22 >= 0 )
    v21 = v22;
  if ( (a1[v21 >> 5] & (0x80000000 >> (v22 % 32))) != 0 )
  {
LABEL_8:
    v26 = result;
    sub_70A250(a1, 1);
    v15 = *v9;
    v16 = a6;
    v17 = (v26 + *v9) & ~(v11 >> 1);
    *v9 = v17;
    if ( v15 > v17 )
    {
      v18 = v8 - 1;
      if ( v18 >= 0 )
      {
        v19 = v18;
        do
        {
          if ( a1[v19]++ != -1 )
            break;
          --v19;
        }
        while ( (int)v19 >= 0 );
      }
    }
    result = a5 == 0 ? 0x80000000 : 0x40000000;
    if ( ((unsigned int)result & *a1) != 0 )
    {
      ++*a2;
    }
    else
    {
      sub_70A210(a1, 1);
      result = v24;
      v16 = a6;
      a1[4] = v24;
    }
    *v16 = 1;
  }
  return result;
}
