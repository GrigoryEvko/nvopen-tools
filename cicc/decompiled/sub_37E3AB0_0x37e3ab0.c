// Function: sub_37E3AB0
// Address: 0x37e3ab0
//
__int64 __fastcall sub_37E3AB0(_DWORD *a1, _DWORD *a2)
{
  unsigned int v2; // eax
  int v3; // eax
  _DWORD *v4; // rax
  __int64 v5; // r9
  __int64 result; // rax
  _DWORD *v7; // rsi
  int v8; // r8d
  unsigned int v9; // edx
  __int64 *v10; // rdi
  _DWORD *v11; // rsi
  unsigned int v12; // edx
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // rdx

  v2 = a2[2] & 0xFFFFFFFE;
  a2[2] = a1[2] & 0xFFFFFFFE | a2[2] & 1;
  a1[2] = v2 | a1[2] & 1;
  v3 = a1[3];
  a1[3] = a2[3];
  a2[3] = v3;
  if ( (a1[2] & 1) == 0 )
  {
    if ( (a2[2] & 1) == 0 )
    {
      v13 = *((_QWORD *)a1 + 2);
      *((_QWORD *)a1 + 2) = *((_QWORD *)a2 + 2);
      v14 = a2[6];
      *((_QWORD *)a2 + 2) = v13;
      result = (unsigned int)a1[6];
      a1[6] = v14;
      a2[6] = result;
      return result;
    }
    v4 = a2;
    a2 = a1;
    a1 = v4;
    goto LABEL_4;
  }
  if ( (a2[2] & 1) == 0 )
  {
LABEL_4:
    *((_BYTE *)a2 + 8) |= 1u;
    v5 = *((_QWORD *)a2 + 2);
    result = (__int64)(a1 + 4);
    v7 = a2 + 4;
    v8 = v7[2];
    do
    {
      v9 = *(_DWORD *)result;
      *v7 = *(_DWORD *)result;
      if ( v9 <= 0xFFFFFFFD )
        *((_QWORD *)v7 + 1) = *(_QWORD *)(result + 8);
      result += 16;
      v7 += 4;
    }
    while ( (_DWORD *)result != a1 + 20 );
    *((_BYTE *)a1 + 8) &= ~1u;
    *((_QWORD *)a1 + 2) = v5;
    a1[6] = v8;
    return result;
  }
  result = (__int64)(a2 + 4);
  v10 = (__int64 *)(a1 + 4);
  v11 = a2 + 20;
  do
  {
    v12 = *(_DWORD *)v10;
    if ( *(_DWORD *)result <= 0xFFFFFFFD )
    {
      if ( v12 > 0xFFFFFFFD )
      {
        *(_DWORD *)v10 = *(_DWORD *)result;
        *(_DWORD *)result = v12;
        v10[1] = *(_QWORD *)(result + 8);
      }
      else
      {
        v15 = *v10;
        v16 = v10[1];
        *(_DWORD *)v10 = *(_DWORD *)result;
        v10[1] = *(_QWORD *)(result + 8);
        *(_DWORD *)result = v15;
        *(_QWORD *)(result + 8) = v16;
      }
    }
    else
    {
      *(_DWORD *)v10 = *(_DWORD *)result;
      *(_DWORD *)result = v12;
      if ( v12 <= 0xFFFFFFFD )
        *(_QWORD *)(result + 8) = v10[1];
    }
    result += 16;
    v10 += 2;
  }
  while ( v11 != (_DWORD *)result );
  return result;
}
