// Function: sub_13FBA80
// Address: 0x13fba80
//
unsigned __int64 __fastcall sub_13FBA80(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  int v3; // r8d
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 v8; // rsi
  __int64 *v9; // r15
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  _QWORD *v13; // rdi
  char *v14; // rax
  char *v15; // rdx
  char *v16; // rsi
  __int64 v17; // r14
  _QWORD *v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // [rsp-50h] [rbp-50h]
  __int64 v22; // [rsp-40h] [rbp-40h] BYREF

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v3 = 1;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v21 = (__int64 *)(v6 + 16LL * v7);
    v8 = *v21;
    if ( a2 == *v21 )
    {
LABEL_3:
      result = v6 + 16 * result;
      if ( v21 == (__int64 *)result )
        return result;
      v9 = (__int64 *)v21[1];
      if ( !v9 )
      {
LABEL_18:
        result = (unsigned __int64)v21;
        *v21 = -16;
        --*(_DWORD *)(a1 + 16);
        ++*(_DWORD *)(a1 + 20);
        return result;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v12 = v9[5];
          v13 = (_QWORD *)v9[4];
          v22 = a2;
          v14 = (char *)sub_13F9BE0(v13, v12, &v22);
          v15 = (char *)v9[5];
          v16 = v14 + 8;
          if ( v15 != v14 + 8 )
          {
            memmove(v14, v16, v15 - v16);
            v16 = (char *)v9[5];
          }
          v17 = v22;
          v10 = (_QWORD *)v9[8];
          v9[5] = (__int64)(v16 - 8);
          if ( (_QWORD *)v9[9] != v10 )
            break;
          v18 = &v10[*((unsigned int *)v9 + 21)];
          if ( v10 == v18 )
          {
LABEL_23:
            v10 = v18;
          }
          else
          {
            while ( v17 != *v10 )
            {
              if ( v18 == ++v10 )
                goto LABEL_23;
            }
          }
LABEL_16:
          if ( v18 == v10 )
            goto LABEL_8;
          *v10 = -2;
          ++*((_DWORD *)v9 + 22);
          v9 = (__int64 *)*v9;
          if ( !v9 )
            goto LABEL_18;
        }
        v10 = (_QWORD *)sub_16CC9F0(v9 + 7, v17);
        if ( v17 == *v10 )
        {
          v19 = v9[9];
          if ( v19 == v9[8] )
            v20 = *((unsigned int *)v9 + 21);
          else
            v20 = *((unsigned int *)v9 + 20);
          v18 = (_QWORD *)(v19 + 8 * v20);
          goto LABEL_16;
        }
        v11 = v9[9];
        if ( v11 == v9[8] )
        {
          v10 = (_QWORD *)(v11 + 8LL * *((unsigned int *)v9 + 21));
          v18 = v10;
          goto LABEL_16;
        }
LABEL_8:
        v9 = (__int64 *)*v9;
        if ( !v9 )
          goto LABEL_18;
      }
    }
    while ( v8 != -8 )
    {
      v7 = (result - 1) & (v3 + v7);
      v21 = (__int64 *)(v6 + 16LL * v7);
      v8 = *v21;
      if ( a2 == *v21 )
        goto LABEL_3;
      ++v3;
    }
  }
  return result;
}
