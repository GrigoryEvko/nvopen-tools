// Function: sub_27ACAE0
// Address: 0x27acae0
//
_QWORD *__fastcall sub_27ACAE0(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  int v4; // ecx
  __int64 v5; // r11
  __int64 v6; // r13
  __int64 v7; // r8
  __int64 v8; // rdi
  unsigned int v10; // ebx
  int v11; // ecx
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // r14
  unsigned int v15; // r10d
  unsigned int v16; // esi
  __int64 *v17; // rdx
  __int64 v18; // r15
  __int64 v19; // rdx
  int v20; // edx
  int v21; // edx
  int v22; // r14d
  int v23; // r10d

  result = a1 - 2;
  v4 = *(_DWORD *)(a2 + 24);
  v5 = *a1;
  v6 = a1[1];
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(a1 - 2);
  if ( v4 )
  {
    v10 = ((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4);
    while ( 1 )
    {
      v11 = v4 - 1;
      v12 = v11 & v10;
      v13 = (__int64 *)(v7 + 16LL * (v11 & v10));
      v14 = *v13;
      if ( v5 == *v13 )
      {
LABEL_5:
        v15 = *((_DWORD *)v13 + 2);
      }
      else
      {
        v21 = 1;
        while ( v14 != -4096 )
        {
          v23 = v21 + 1;
          v12 = v11 & (v21 + v12);
          v13 = (__int64 *)(v7 + 16LL * v12);
          v14 = *v13;
          if ( v5 == *v13 )
            goto LABEL_5;
          v21 = v23;
        }
        v15 = 0;
      }
      v16 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v17 = (__int64 *)(v7 + 16LL * v16);
      v18 = *v17;
      if ( v8 != *v17 )
        break;
LABEL_7:
      if ( v15 >= *((_DWORD *)v17 + 2) )
        goto LABEL_12;
      v19 = result[1];
      result[2] = v8;
      v8 = *(result - 2);
      result[3] = v19;
      v4 = *(_DWORD *)(a2 + 24);
      v7 = *(_QWORD *)(a2 + 8);
      if ( !v4 )
        goto LABEL_9;
      result -= 2;
    }
    v20 = 1;
    while ( v18 != -4096 )
    {
      v22 = v20 + 1;
      v16 = v11 & (v20 + v16);
      v17 = (__int64 *)(v7 + 16LL * v16);
      v18 = *v17;
      if ( v8 == *v17 )
        goto LABEL_7;
      v20 = v22;
    }
LABEL_12:
    result += 2;
  }
  else
  {
    result = a1;
  }
LABEL_9:
  *result = v5;
  result[1] = v6;
  return result;
}
