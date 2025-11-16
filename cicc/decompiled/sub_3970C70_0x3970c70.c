// Function: sub_3970C70
// Address: 0x3970c70
//
__int64 __fastcall sub_3970C70(__int64 *a1, _QWORD *a2, bool *a3)
{
  __int64 v5; // rax
  bool v7; // al
  char *v8; // rsi
  unsigned __int64 v9; // rdx
  char *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdi
  char *v13; // rcx
  bool v14; // cl
  char v15; // dl
  __int64 result; // rax
  __int64 v17; // r14

  v5 = a1[33];
  v7 = !*(_BYTE *)(v5 + 345)
    && (v17 = *(_QWORD *)v5 + 112LL, !(unsigned __int8)sub_1560180(v17, 34))
    && !(unsigned __int8)sub_1560180(v17, 17)
    && (unsigned int)sub_1700720(a1[29]) != 0;
  *a3 = v7;
  v8 = (char *)a2[9];
  if ( a2 == *(_QWORD **)(a2[7] + 328LL) )
    v9 = 0;
  else
    v9 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = (char *)a2[8];
  v11 = (v8 - v10) >> 5;
  v12 = (v8 - v10) >> 3;
  if ( v11 > 0 )
  {
    v13 = &v10[32 * v11];
    while ( v9 != *(_QWORD *)v10 )
    {
      if ( v9 == *((_QWORD *)v10 + 1) )
      {
        v14 = v8 != v10 + 8;
        goto LABEL_13;
      }
      if ( v9 == *((_QWORD *)v10 + 2) )
      {
        v14 = v8 != v10 + 16;
        goto LABEL_13;
      }
      if ( v9 == *((_QWORD *)v10 + 3) )
      {
        v14 = v8 != v10 + 24;
        goto LABEL_13;
      }
      v10 += 32;
      if ( v13 == v10 )
      {
        v12 = (v8 - v10) >> 3;
        goto LABEL_20;
      }
    }
    goto LABEL_12;
  }
LABEL_20:
  if ( v12 != 2 )
  {
    if ( v12 != 3 )
    {
      v14 = 0;
      if ( v12 != 1 )
        goto LABEL_13;
      goto LABEL_23;
    }
    if ( v9 == *(_QWORD *)v10 )
    {
LABEL_12:
      v14 = v8 != v10;
      goto LABEL_13;
    }
    v10 += 8;
  }
  if ( v9 == *(_QWORD *)v10 )
    goto LABEL_12;
  v10 += 8;
LABEL_23:
  v14 = 0;
  if ( v9 == *(_QWORD *)v10 )
    goto LABEL_12;
LABEL_13:
  a3[1] = v14;
  v15 = 0;
  result = (__int64)(a2[9] - a2[8]) >> 3;
  if ( (_DWORD)result )
  {
    result = (*(unsigned int (__fastcall **)(__int64 *, _QWORD *, _QWORD))(*a1 + 320))(a1, a2, 0) ^ 1;
    v15 = result;
  }
  a3[2] = v15;
  return result;
}
