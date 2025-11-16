// Function: sub_30FBA00
// Address: 0x30fba00
//
__int64 __fastcall sub_30FBA00(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __int64 result; // rax
  int v5; // eax
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r9
  unsigned __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *v11; // r8
  _QWORD *v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rcx
  int v15; // edx
  int v16; // r10d

  v2 = a1[21];
  v3 = *(_QWORD *)(v2 + 104);
  result = *(unsigned int *)(v2 + 120);
  if ( !(_DWORD)result )
    return result;
  v5 = result - 1;
  v6 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v3 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v15 = 1;
    while ( v8 != -4096 )
    {
      v16 = v15 + 1;
      v6 = v5 & (v15 + v6);
      v7 = (__int64 *)(v3 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v15 = v16;
    }
    return 0;
  }
LABEL_3:
  v9 = v7[1];
  if ( !v9 )
    return 0;
  v10 = (_QWORD *)a1[27];
  v11 = a1 + 26;
  if ( !v10 )
    goto LABEL_16;
  v12 = a1 + 26;
  do
  {
    while ( 1 )
    {
      v13 = v10[2];
      v14 = v10[3];
      if ( v10[4] >= v9 )
        break;
      v10 = (_QWORD *)v10[3];
      if ( !v14 )
        goto LABEL_9;
    }
    v12 = v10;
    v10 = (_QWORD *)v10[2];
  }
  while ( v13 );
LABEL_9:
  if ( v11 == v12 || v12[4] > v9 )
LABEL_16:
    sub_426320((__int64)"map::at");
  return *((unsigned int *)v12 + 10);
}
