// Function: sub_13E5870
// Address: 0x13e5870
//
char *__fastcall sub_13E5870(_QWORD *a1, char **a2, unsigned __int64 a3)
{
  _QWORD *v5; // rsi
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // r12
  char *v10; // r13
  __int64 v11; // rax
  char *v12; // r14
  __int64 v13; // rcx
  char *v14; // rax
  __int64 v15; // rax
  char *v16; // rsi
  char *result; // rax
  char *v18; // rdx
  _QWORD *v19; // r14
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rdx
  _QWORD *v23; // rax
  _BOOL4 v24; // r12d
  __int64 v25; // rax
  char *v26; // rdi
  char *v27; // r13
  char *v28; // rax
  char *v29; // rax
  char *v30; // r15
  int v31; // r14d
  unsigned int v32; // r13d
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // [rsp+8h] [rbp-58h]
  _QWORD *v38; // [rsp+18h] [rbp-48h]
  unsigned __int64 v39; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v40[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a1 + 4;
  v6 = (_QWORD *)a1[5];
  v39 = a3;
  v38 = a1 + 4;
  if ( !v6 )
    goto LABEL_8;
  do
  {
    while ( 1 )
    {
      v7 = v6[2];
      v8 = v6[3];
      if ( v6[4] >= a3 )
        break;
      v6 = (_QWORD *)v6[3];
      if ( !v8 )
        goto LABEL_6;
    }
    v5 = v6;
    v6 = (_QWORD *)v6[2];
  }
  while ( v7 );
LABEL_6:
  if ( v38 == v5 || v5[4] > a3 )
  {
LABEL_8:
    v9 = *(_QWORD *)(a3 + 8);
    v10 = a2[2];
    if ( !v9 )
      goto LABEL_28;
    while ( 1 )
    {
      v11 = sub_1648700(v9);
      if ( (unsigned __int8)(*(_BYTE *)(v11 + 16) - 25) <= 9u )
        break;
      v9 = *(_QWORD *)(v9 + 8);
      if ( !v9 )
        goto LABEL_28;
    }
    v12 = a2[1];
    v13 = *(_QWORD *)(v11 + 40);
    if ( v12 != v10 )
    {
LABEL_11:
      v14 = v12;
      while ( *(_QWORD *)v14 != v13 )
      {
        v14 += 8;
        if ( v10 == v14 )
          goto LABEL_17;
      }
      while ( 1 )
      {
        v9 = *(_QWORD *)(v9 + 8);
        if ( !v9 )
          break;
        v15 = sub_1648700(v9);
        if ( (unsigned __int8)(*(_BYTE *)(v15 + 16) - 25) <= 9u )
        {
          v13 = *(_QWORD *)(v15 + 40);
          if ( v12 != v10 )
            goto LABEL_11;
          goto LABEL_17;
        }
      }
LABEL_28:
      v40[0] = a3;
      if ( v10 == a2[3] )
      {
        sub_1292090((__int64)(a2 + 1), v10, v40);
      }
      else
      {
        if ( v10 )
        {
          *(_QWORD *)v10 = a3;
          v10 = a2[2];
        }
        a2[2] = v10 + 8;
      }
      v19 = (_QWORD *)a1[5];
      if ( v19 )
      {
        v20 = v39;
        v21 = v39;
        while ( 1 )
        {
          v22 = v19[4];
          v23 = (_QWORD *)v19[3];
          if ( v39 < v22 )
            v23 = (_QWORD *)v19[2];
          if ( !v23 )
            break;
          v19 = v23;
        }
        if ( v39 >= v22 )
        {
          if ( v39 <= v22 )
          {
LABEL_43:
            v26 = a2[4];
            v27 = a2[5];
            if ( v26 != v27 )
            {
              v28 = a2[4];
              while ( *(_QWORD *)v28 != v21 )
              {
                v28 += 8;
                if ( v27 == v28 )
                  goto LABEL_48;
              }
              v29 = sub_13E5700(v26, a2[5], (__int64 *)&v39);
              sub_13E5810((__int64)(a2 + 4), v29, v27);
            }
LABEL_48:
            result = (char *)sub_157EBA0(a3);
            v30 = result;
            if ( result )
            {
              result = (char *)sub_15F4D60(result);
              v31 = (int)result;
              if ( (_DWORD)result )
              {
                v32 = 0;
                do
                {
                  v33 = v32++;
                  v34 = sub_15F4DF0(v30, v33);
                  result = (char *)sub_13E5870(a1, a2, v34);
                }
                while ( v32 != v31 );
              }
            }
            return result;
          }
          goto LABEL_40;
        }
        if ( (_QWORD *)a1[6] == v19 )
        {
LABEL_40:
          v24 = 1;
          if ( v38 != v19 )
            v24 = v20 < v19[4];
          goto LABEL_42;
        }
      }
      else
      {
        v19 = a1 + 4;
        if ( v38 == (_QWORD *)a1[6] )
        {
          v19 = a1 + 4;
          v24 = 1;
LABEL_42:
          v25 = sub_22077B0(40);
          *(_QWORD *)(v25 + 32) = v39;
          sub_220F040(v24, v25, v19, v38);
          v21 = v39;
          ++a1[8];
          goto LABEL_43;
        }
        v20 = v39;
      }
      v36 = v20;
      v35 = sub_220EF80(v19);
      v20 = v36;
      v21 = v36;
      if ( *(_QWORD *)(v35 + 32) >= v36 )
        goto LABEL_43;
      goto LABEL_40;
    }
LABEL_17:
    v16 = a2[5];
    result = a2[4];
    if ( result != v16 )
    {
      while ( a3 != *(_QWORD *)result )
      {
        result += 8;
        if ( v16 == result )
          goto LABEL_58;
      }
      return result;
    }
LABEL_58:
    if ( v16 == a2[6] )
      return sub_1292090((__int64)(a2 + 4), v16, &v39);
    if ( v16 )
    {
      *(_QWORD *)v16 = a3;
      v16 = a2[5];
    }
    a2[5] = v16 + 8;
    return result;
  }
  v18 = a2[2];
  result = a2[1];
  if ( result == v18 )
  {
LABEL_53:
    v16 = a2[5];
    result = a2[4];
    if ( result == v16 )
      goto LABEL_58;
    while ( a3 != *(_QWORD *)result )
    {
      result += 8;
      if ( v16 == result )
        goto LABEL_58;
    }
  }
  else
  {
    while ( a3 != *(_QWORD *)result )
    {
      result += 8;
      if ( v18 == result )
        goto LABEL_53;
    }
  }
  return result;
}
