// Function: sub_D9B980
// Address: 0xd9b980
//
char *__fastcall sub_D9B980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  char *result; // rax
  char *v12; // r12
  _QWORD *v13; // rdx
  char *v14; // rbx
  char v15; // di
  _QWORD *v16; // rax
  _BYTE v17[8]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v18; // [rsp+8h] [rbp-68h]
  char *v19; // [rsp+10h] [rbp-60h]
  __int64 v20; // [rsp+18h] [rbp-58h]
  int v21; // [rsp+20h] [rbp-50h]
  unsigned __int8 v22; // [rsp+24h] [rbp-4Ch]
  char v23; // [rsp+28h] [rbp-48h] BYREF

  v7 = v17;
  v19 = &v23;
  v17[0] = 0;
  v18 = 0;
  v20 = 4;
  v21 = 0;
  v22 = 1;
  sub_D9B3F0(a3, (__int64)v17, a3, a4, a5, a6);
  v10 = v22;
  result = v19;
  if ( v22 )
  {
    v12 = &v19[8 * HIDWORD(v20)];
    if ( v19 != v12 )
      goto LABEL_3;
  }
  else
  {
    v12 = &v19[8 * (unsigned int)v20];
    if ( v19 != v12 )
    {
LABEL_3:
      while ( 1 )
      {
        v13 = *(_QWORD **)result;
        v14 = result;
        if ( *(_QWORD *)result < 0xFFFFFFFFFFFFFFFELL )
          break;
        result += 8;
        if ( v12 == result )
          goto LABEL_5;
      }
      if ( result == v12 )
      {
LABEL_5:
        if ( !v22 )
          return (char *)_libc_free(v19, v7);
        return result;
      }
      v15 = *(_BYTE *)(a2 + 28);
      v7 = (_BYTE *)v13[3];
      if ( !v15 )
        goto LABEL_20;
LABEL_9:
      v16 = *(_QWORD **)(a2 + 8);
      v10 = *(unsigned int *)(a2 + 20);
      v13 = &v16[v10];
      if ( v16 == v13 )
      {
LABEL_21:
        if ( (unsigned int)v10 < *(_DWORD *)(a2 + 16) )
        {
          v10 = (unsigned int)(v10 + 1);
          *(_DWORD *)(a2 + 20) = v10;
          *v13 = v7;
          v15 = *(_BYTE *)(a2 + 28);
          ++*(_QWORD *)a2;
          goto LABEL_13;
        }
        goto LABEL_20;
      }
      while ( v7 != (_BYTE *)*v16 )
      {
        if ( v13 == ++v16 )
          goto LABEL_21;
      }
LABEL_13:
      while ( 1 )
      {
        result = v14 + 8;
        if ( v14 + 8 == v12 )
          break;
        while ( 1 )
        {
          v13 = *(_QWORD **)result;
          v14 = result;
          if ( *(_QWORD *)result < 0xFFFFFFFFFFFFFFFELL )
            break;
          result += 8;
          if ( v12 == result )
            goto LABEL_16;
        }
        if ( result == v12 )
          break;
        v7 = (_BYTE *)v13[3];
        if ( v15 )
          goto LABEL_9;
LABEL_20:
        sub_C8CC70(a2, (__int64)v7, (__int64)v13, v10, v8, v9);
        v15 = *(_BYTE *)(a2 + 28);
      }
LABEL_16:
      if ( v22 )
        return result;
    }
    return (char *)_libc_free(v19, v7);
  }
  return result;
}
