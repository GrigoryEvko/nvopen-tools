// Function: sub_2AE9220
// Address: 0x2ae9220
//
__int64 __fastcall sub_2AE9220(_DWORD *a1, _DWORD *a2)
{
  unsigned int v2; // eax
  int v3; // eax
  _DWORD *v4; // rax
  __int64 v5; // r8
  __int64 result; // rax
  int v7; // ecx
  unsigned int v8; // edx
  unsigned int *v9; // rsi
  _DWORD *v10; // rdi
  unsigned int v11; // edx
  unsigned int v12; // ecx
  unsigned int v13; // edx
  __int64 v14; // rax
  int v15; // edx
  unsigned int v16; // r8d

  v2 = a2[2] & 0xFFFFFFFE;
  a2[2] = a1[2] & 0xFFFFFFFE | a2[2] & 1;
  a1[2] = v2 | a1[2] & 1;
  v3 = a1[3];
  a1[3] = a2[3];
  a2[3] = v3;
  if ( (a1[2] & 1) != 0 )
  {
    if ( (a2[2] & 1) == 0 )
      goto LABEL_4;
    result = (__int64)(a1 + 4);
    v9 = a2 + 4;
    v10 = a1 + 12;
    while ( 1 )
    {
      v11 = *v9;
      v12 = *(_DWORD *)result;
      if ( *v9 == -1 )
      {
        *(_DWORD *)result = -1;
        *v9 = v12;
        if ( v12 <= 0xFFFFFFFD )
          goto LABEL_20;
      }
      else
      {
        if ( v11 != -2 )
        {
          *(_DWORD *)result = v11;
          if ( v12 <= 0xFFFFFFFD )
          {
            v16 = *(_DWORD *)(result + 4);
            *(_DWORD *)(result + 4) = v9[1];
            *v9 = v12;
            v9[1] = v16;
          }
          else
          {
            v13 = v9[1];
            *v9 = v12;
            *(_DWORD *)(result + 4) = v13;
          }
          goto LABEL_15;
        }
        *(_DWORD *)result = -2;
        *v9 = v12;
        if ( v12 <= 0xFFFFFFFD )
LABEL_20:
          v9[1] = *(_DWORD *)(result + 4);
      }
LABEL_15:
      result += 8;
      v9 += 2;
      if ( v10 == (_DWORD *)result )
        return result;
    }
  }
  if ( (a2[2] & 1) == 0 )
  {
    v14 = *((_QWORD *)a1 + 2);
    *((_QWORD *)a1 + 2) = *((_QWORD *)a2 + 2);
    v15 = a2[6];
    *((_QWORD *)a2 + 2) = v14;
    result = (unsigned int)a1[6];
    a1[6] = v15;
    a2[6] = result;
    return result;
  }
  v4 = a2;
  a2 = a1;
  a1 = v4;
LABEL_4:
  *((_BYTE *)a2 + 8) |= 1u;
  v5 = *((_QWORD *)a2 + 2);
  result = 16;
  v7 = a2[6];
  do
  {
    v8 = *(_DWORD *)((char *)a1 + result);
    *(_DWORD *)((char *)a2 + result) = v8;
    if ( v8 <= 0xFFFFFFFD )
      *(_DWORD *)((char *)a2 + result + 4) = *(_DWORD *)((char *)a1 + result + 4);
    result += 8;
  }
  while ( result != 48 );
  *((_BYTE *)a1 + 8) &= ~1u;
  *((_QWORD *)a1 + 2) = v5;
  a1[6] = v7;
  return result;
}
