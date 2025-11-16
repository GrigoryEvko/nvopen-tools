// Function: sub_6307F0
// Address: 0x6307f0
//
__int64 __fastcall sub_6307F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  unsigned __int8 v5; // cl
  unsigned __int8 v6; // si
  __int64 *v7; // rax
  __int64 v8; // r8
  _QWORD *v9; // rax

  result = *(unsigned int *)(a2 + 84);
  if ( !(_DWORD)result )
  {
    v4 = *(_QWORD *)(a2 + 56);
    if ( v4 )
    {
      v5 = *(_BYTE *)(a1 + 8);
      v6 = *(_BYTE *)(v4 + 8);
      if ( v5 < v6 )
      {
LABEL_8:
        v7 = *(__int64 **)(v4 + 16);
        if ( v6 != 2 )
          v7 = (__int64 *)v7[5];
        v8 = *v7;
        v9 = *(_QWORD **)(a1 + 16);
        if ( v5 != 2 )
          v9 = (_QWORD *)v9[5];
        result = sub_686B60(4, 1719, dword_4F07508, *v9, v8);
        *(_DWORD *)(a2 + 84) = 1;
      }
      else if ( v5 == v6 )
      {
        result = v4;
        while ( a1 != result )
        {
          result = *(_QWORD *)result;
          if ( !result )
            goto LABEL_8;
        }
      }
    }
  }
  *(_QWORD *)(a2 + 56) = a1;
  return result;
}
