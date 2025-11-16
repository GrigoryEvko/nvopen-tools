// Function: sub_878E70
// Address: 0x878e70
//
__int64 __fastcall sub_878E70(__int64 a1, unsigned int a2, unsigned __int64 a3, char a4, char a5)
{
  char v7; // bl
  __int64 v8; // r15
  __int64 v9; // rax
  char v10; // cl
  _QWORD *v11; // rdx
  __int64 result; // rax
  __int64 v13; // rcx
  __int64 v14; // rsi

  v7 = a4 & 1 | (2 * (a5 & 1));
  v8 = sub_8661A0(1);
  v9 = sub_823970(48);
  v10 = *(_BYTE *)(v9 + 40);
  *(_QWORD *)(v9 + 8) = 0;
  v11 = (_QWORD *)v9;
  *(_QWORD *)v9 = 0;
  *(_QWORD *)(v9 + 32) = a1;
  *(_DWORD *)(v9 + 16) = a2;
  *(_QWORD *)(v9 + 24) = a3;
  *(_BYTE *)(v9 + 40) = v10 & 0xFC | v7;
  result = *(_QWORD *)(v8 + 56);
  if ( !result )
    goto LABEL_9;
  if ( *(_DWORD *)(result + 16) > a2 || *(_DWORD *)(result + 16) == a2 && *(_QWORD *)(result + 24) > a3 )
  {
    v11[1] = result;
    *(_QWORD *)result = v11;
LABEL_9:
    *(_QWORD *)(v8 + 56) = v11;
    goto LABEL_10;
  }
  v13 = *(_QWORD *)(v8 + 64);
  if ( *(_DWORD *)(v13 + 16) <= a2 )
  {
    if ( *(_DWORD *)(v13 + 16) == a2 )
    {
      if ( *(_QWORD *)(v13 + 24) <= a3 )
        result = *(_QWORD *)(v8 + 64);
    }
    else
    {
      result = *(_QWORD *)(v8 + 64);
    }
  }
  while ( 1 )
  {
    v14 = result;
    result = *(_QWORD *)(result + 8);
    if ( !result )
      break;
    if ( *(_DWORD *)(result + 16) >= a2 )
    {
      while ( *(_DWORD *)(result + 16) == a2 && *(_QWORD *)(result + 24) < a3 )
      {
        v14 = result;
        if ( !*(_QWORD *)(result + 8) )
        {
          result = 0;
          goto LABEL_13;
        }
        result = *(_QWORD *)(result + 8);
      }
      break;
    }
  }
LABEL_13:
  v11[1] = result;
  *v11 = v14;
  result = *(_QWORD *)(v14 + 8);
  if ( result )
    *(_QWORD *)result = v11;
  *(_QWORD *)(v14 + 8) = v11;
LABEL_10:
  *(_QWORD *)(v8 + 64) = v11;
  return result;
}
