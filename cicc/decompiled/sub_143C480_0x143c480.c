// Function: sub_143C480
// Address: 0x143c480
//
char __fastcall sub_143C480(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v5; // rax
  __int64 v8; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdx
  char result; // al
  __int64 v19; // rax
  int v20; // edx
  int v21; // r10d

  if ( !a4 )
    goto LABEL_10;
  v5 = *(unsigned int *)(a4 + 48);
  if ( !(_DWORD)v5 )
    goto LABEL_10;
  v8 = *(_QWORD *)(a4 + 32);
  v11 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v12 = (__int64 *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( a3 != *v12 )
  {
    v20 = 1;
    while ( v13 != -8 )
    {
      v21 = v20 + 1;
      v11 = (v5 - 1) & (v20 + v11);
      v12 = (__int64 *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( a3 == *v12 )
        goto LABEL_4;
      v20 = v21;
    }
    goto LABEL_10;
  }
LABEL_4:
  if ( v12 == (__int64 *)(v8 + 16 * v5) || !v12[1] )
  {
LABEL_10:
    *a1 = 0;
    return 1;
  }
  if ( !a5 )
  {
    v19 = sub_143B970((__int64)a1, *a1, a2, a3, 0);
    *a1 = v19;
    return v19 == 0;
  }
  v14 = sub_143B970((__int64)a1, *a1, a2, a3, a4);
  *a1 = v14;
  v17 = v14;
  if ( !v14 )
    return a5;
  result = 0;
  if ( *(_BYTE *)(v17 + 16) > 0x17u )
  {
    if ( (unsigned __int8)sub_15CC8F0(a4, *(_QWORD *)(v17 + 40), a3, v15, v16) )
      return *a1 == 0;
    goto LABEL_10;
  }
  return result;
}
