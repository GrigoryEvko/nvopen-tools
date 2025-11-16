// Function: sub_1E89640
// Address: 0x1e89640
//
_DWORD *__fastcall sub_1E89640(__int64 a1, __int64 a2)
{
  _DWORD *result; // rax
  __int64 v3; // r11
  int v4; // edx
  int v5; // ecx
  __int64 v6; // r10
  unsigned int v7; // edx
  int *v8; // r8
  int v9; // r9d
  int v10; // r8d
  int v11; // ebx

  result = *(_DWORD **)a2;
  v3 = *(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8);
  if ( v3 != *(_QWORD *)a2 )
  {
    while ( 1 )
    {
      v4 = *(_DWORD *)(a1 + 24);
      if ( v4 )
        break;
      if ( ++result == (_DWORD *)v3 )
        return result;
    }
LABEL_3:
    v5 = v4 - 1;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = (v4 - 1) & (37 * *result);
    v8 = (int *)(v6 + 4LL * (v5 & (unsigned int)(37 * *result)));
    v9 = *v8;
    if ( *result == *v8 )
    {
LABEL_4:
      *v8 = -2;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
      goto LABEL_5;
    }
    v10 = 1;
    while ( v9 != -1 )
    {
      v11 = v10 + 1;
      v7 = v5 & (v10 + v7);
      v8 = (int *)(v6 + 4LL * v7);
      v9 = *v8;
      if ( *result == *v8 )
        goto LABEL_4;
      v10 = v11;
    }
LABEL_5:
    while ( ++result != (_DWORD *)v3 )
    {
      v4 = *(_DWORD *)(a1 + 24);
      if ( v4 )
        goto LABEL_3;
    }
  }
  return result;
}
