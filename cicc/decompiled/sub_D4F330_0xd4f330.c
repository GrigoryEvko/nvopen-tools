// Function: sub_D4F330
// Address: 0xd4f330
//
__int64 *__fastcall sub_D4F330(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // r9
  __int64 v8; // r8
  int v9; // r11d
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 *v12; // rax
  __int64 v13; // r10
  _QWORD *v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // r8
  __int64 v17; // r8
  __int64 *result; // rax
  int v19; // eax
  int v20; // eax
  __int64 v21; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v22[5]; // [rsp+18h] [rbp-28h] BYREF

  v21 = a2;
  v5 = *(_DWORD *)(a3 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a3;
    v22[0] = 0;
    goto LABEL_36;
  }
  v6 = v21;
  v7 = v5 - 1;
  v8 = *(_QWORD *)(a3 + 8);
  v9 = 1;
  v10 = 0;
  v11 = (unsigned int)v7 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v12 = (__int64 *)(v8 + 16 * v11);
  v13 = *v12;
  if ( v21 != *v12 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v10 )
        v10 = v12;
      v11 = (unsigned int)v7 & (v9 + (_DWORD)v11);
      v12 = (__int64 *)(v8 + 16LL * (unsigned int)v11);
      v13 = *v12;
      if ( v21 == *v12 )
        goto LABEL_3;
      ++v9;
    }
    if ( !v10 )
      v10 = v12;
    v19 = *(_DWORD *)(a3 + 16);
    ++*(_QWORD *)a3;
    v20 = v19 + 1;
    v22[0] = v10;
    if ( 4 * v20 < 3 * v5 )
    {
      v11 = v5 - *(_DWORD *)(a3 + 20) - v20;
      if ( (unsigned int)v11 > v5 >> 3 )
      {
LABEL_32:
        *(_DWORD *)(a3 + 16) = v20;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v10 = v6;
        v14 = v10 + 1;
        v10[1] = 0;
        goto LABEL_4;
      }
LABEL_37:
      sub_D4F150(a3, v5);
      sub_D4C730(a3, &v21, v22);
      v6 = v21;
      v10 = (__int64 *)v22[0];
      v20 = *(_DWORD *)(a3 + 16) + 1;
      goto LABEL_32;
    }
LABEL_36:
    v5 *= 2;
    goto LABEL_37;
  }
LABEL_3:
  v14 = v12 + 1;
LABEL_4:
  *v14 = a1;
  do
  {
    while ( 1 )
    {
      v15 = v21;
      v22[0] = v21;
      v16 = (_QWORD *)a1[5];
      if ( v16 == (_QWORD *)a1[6] )
      {
        sub_9319A0((__int64)(a1 + 4), (_BYTE *)a1[5], v22);
        v15 = v22[0];
      }
      else
      {
        if ( v16 )
        {
          *v16 = v21;
          v16 = (_QWORD *)a1[5];
        }
        v17 = (__int64)(v16 + 1);
        a1[5] = v17;
      }
      if ( !*((_BYTE *)a1 + 84) )
        goto LABEL_16;
      result = (__int64 *)a1[8];
      v11 = *((unsigned int *)a1 + 19);
      v10 = &result[v11];
      if ( result != v10 )
        break;
LABEL_18:
      if ( (unsigned int)v11 >= *((_DWORD *)a1 + 18) )
      {
LABEL_16:
        result = sub_C8CC70((__int64)(a1 + 7), v15, (__int64)v10, v11, v17, v7);
        a1 = (__int64 *)*a1;
        if ( !a1 )
          return result;
      }
      else
      {
        v11 = (unsigned int)(v11 + 1);
        *((_DWORD *)a1 + 19) = v11;
        *v10 = v15;
        ++a1[7];
        a1 = (__int64 *)*a1;
        if ( !a1 )
          return result;
      }
    }
    while ( v15 != *result )
    {
      if ( v10 == ++result )
        goto LABEL_18;
    }
    a1 = (__int64 *)*a1;
  }
  while ( a1 );
  return result;
}
