// Function: sub_1C44570
// Address: 0x1c44570
//
__int64 __fastcall sub_1C44570(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // r12
  __int64 result; // rax
  unsigned __int64 v4; // rbx
  _QWORD *v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // r14
  _BOOL4 v9; // ebx
  __int64 v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // rsi
  unsigned __int64 v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a1 + 25;
  result = a1[26];
  v15[0] = a2;
  if ( !result )
    goto LABEL_8;
  v4 = a2;
  v5 = a1 + 25;
  do
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(result + 16);
      v7 = *(_QWORD *)(result + 24);
      if ( *(_QWORD *)(result + 32) >= v4 )
        break;
      result = *(_QWORD *)(result + 24);
      if ( !v7 )
        goto LABEL_6;
    }
    v5 = (_QWORD *)result;
    result = *(_QWORD *)(result + 16);
  }
  while ( v6 );
LABEL_6:
  if ( v2 == v5 || v5[4] > v4 )
  {
LABEL_8:
    result = (__int64)sub_1C444D0((__int64)(a1 + 24), v15);
    v8 = v7;
    if ( v7 )
    {
      v9 = result || v2 == (_QWORD *)v7 || v15[0] < *(_QWORD *)(v7 + 32);
      v10 = sub_22077B0(40);
      *(_QWORD *)(v10 + 32) = v15[0];
      result = sub_220F040(v9, v10, v8, v2);
      ++a1[29];
    }
    v11 = (_BYTE *)a1[22];
    if ( v11 == (_BYTE *)a1[23] )
    {
      result = (__int64)sub_170B610((__int64)(a1 + 21), v11, v15);
      v4 = v15[0];
    }
    else
    {
      v4 = v15[0];
      if ( v11 )
      {
        *(_QWORD *)v11 = v15[0];
        v11 = (_BYTE *)a1[22];
        v4 = v15[0];
      }
      a1[22] = v11 + 8;
    }
  }
  if ( *(_BYTE *)(v4 + 16) == 77 )
  {
    result = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
    if ( (*(_DWORD *)(v4 + 20) & 0xFFFFFFF) != 0 )
    {
      v12 = 0;
      v13 = 24LL * (unsigned int)result;
      do
      {
        if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
        {
          result = *(_QWORD *)(v4 - 8);
        }
        else
        {
          v7 = 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
          result = v4 - v7;
        }
        v14 = *(_QWORD *)(result + v12);
        if ( *(_BYTE *)(v14 + 16) > 0x17u )
          result = sub_1C44570(a1, v14, v7, v6);
        v12 += 24;
      }
      while ( v13 != v12 );
    }
  }
  return result;
}
