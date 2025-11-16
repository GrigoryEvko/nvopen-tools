// Function: sub_23C6E80
// Address: 0x23c6e80
//
__int64 __fastcall sub_23C6E80(__int64 **a1, char *a2, unsigned __int64 a3)
{
  int v5; // ecx
  __int64 *v6; // rsi
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // r15
  __int64 *v11; // r13
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r13
  __int64 *v17; // r14

  v5 = *((_DWORD *)a1 + 2);
  if ( !v5 )
    goto LABEL_16;
  v6 = *a1;
  v7 = **a1;
  if ( v7 && v7 != -8 )
  {
    v10 = *a1;
  }
  else
  {
    v8 = v6 + 1;
    do
    {
      do
      {
        v9 = *v8;
        v10 = v8++;
      }
      while ( !v9 );
    }
    while ( v9 == -8 );
  }
  v11 = &v6[v5];
LABEL_8:
  if ( v10 == v11 )
  {
LABEL_16:
    v16 = a1[3];
    v17 = a1[4];
    if ( v17 == v16 )
    {
      return 0;
    }
    else
    {
      while ( !(unsigned __int8)sub_C89090((_QWORD *)*v16, a2, a3, 0, 0) )
      {
        v16 += 2;
        if ( v17 == v16 )
          return 0;
      }
      return *((unsigned int *)v16 + 2);
    }
  }
  else
  {
    while ( 1 )
    {
      v12 = *v10;
      if ( sub_1099960(*v10 + 8, a2, a3) )
        return *(unsigned int *)(v12 + 80);
      v13 = v10[1];
      if ( v13 && v13 != -8 )
      {
        ++v10;
        goto LABEL_8;
      }
      v14 = v10 + 2;
      do
      {
        do
        {
          v15 = *v14;
          v10 = v14++;
        }
        while ( v15 == -8 );
      }
      while ( !v15 );
      if ( v10 == v11 )
        goto LABEL_16;
    }
  }
}
