// Function: sub_37E3E60
// Address: 0x37e3e60
//
__int64 __fastcall sub_37E3E60(_DWORD *a1, _DWORD *a2)
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
  __int64 v11; // rdx
  unsigned int v12; // edx
  unsigned int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // r8

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
    v10 = a1 + 36;
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *v9;
        v13 = *(_DWORD *)result;
        if ( *v9 == -1 )
          break;
        if ( v12 == -2 )
        {
          *(_DWORD *)result = -2;
          *v9 = v13;
          if ( v13 <= 0xFFFFFFFD )
            goto LABEL_17;
        }
        else
        {
          *(_DWORD *)result = v12;
          if ( v13 <= 0xFFFFFFFD )
          {
            v17 = *(_QWORD *)(result + 8);
            *(_QWORD *)(result + 8) = *((_QWORD *)v9 + 1);
            *v9 = v13;
            *((_QWORD *)v9 + 1) = v17;
          }
          else
          {
            v11 = *((_QWORD *)v9 + 1);
            *v9 = v13;
            *(_QWORD *)(result + 8) = v11;
          }
        }
LABEL_14:
        result += 16;
        v9 += 4;
        if ( v10 == (_DWORD *)result )
          return result;
      }
      *(_DWORD *)result = -1;
      *v9 = v13;
      if ( v13 > 0xFFFFFFFD )
        goto LABEL_14;
LABEL_17:
      v14 = *(_QWORD *)(result + 8);
      result += 16;
      v9 += 4;
      *((_QWORD *)v9 - 1) = v14;
      if ( v10 == (_DWORD *)result )
        return result;
    }
  }
  if ( (a2[2] & 1) == 0 )
  {
    v15 = *((_QWORD *)a1 + 2);
    *((_QWORD *)a1 + 2) = *((_QWORD *)a2 + 2);
    v16 = a2[6];
    *((_QWORD *)a2 + 2) = v15;
    result = (unsigned int)a1[6];
    a1[6] = v16;
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
      *(_QWORD *)((char *)a2 + result + 8) = *(_QWORD *)((char *)a1 + result + 8);
    result += 16;
  }
  while ( result != 144 );
  *((_BYTE *)a1 + 8) &= ~1u;
  *((_QWORD *)a1 + 2) = v5;
  a1[6] = v7;
  return result;
}
