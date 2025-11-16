// Function: sub_1C99380
// Address: 0x1c99380
//
__int64 __fastcall sub_1C99380(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  char v3; // dl
  _BYTE *v4; // rsi
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned int v8; // r12d
  _BYTE *v9; // rsi
  _BYTE *v10; // rsi
  _QWORD v11[2]; // [rsp+8h] [rbp-38h] BYREF
  unsigned int v12; // [rsp+1Ch] [rbp-24h] BYREF
  __int64 v13; // [rsp+20h] [rbp-20h] BYREF
  __int64 v14[3]; // [rsp+28h] [rbp-18h] BYREF

  result = a2;
  v3 = *(_BYTE *)(a2 + 16);
  v11[0] = a2;
  if ( v3 == 75 )
  {
    v13 = a2;
    if ( *(_BYTE *)(**(_QWORD **)(a2 - 48) + 8LL) == 15 && *(_BYTE *)(**(_QWORD **)(a2 - 24) + 8LL) == 15 )
    {
      v4 = (_BYTE *)a1[10];
      if ( v4 == (_BYTE *)a1[11] )
      {
        return (__int64)sub_1C99060((__int64)(a1 + 9), v4, &v13);
      }
      else
      {
        if ( v4 )
        {
          *(_QWORD *)v4 = result;
          v4 = (_BYTE *)a1[10];
        }
        a1[10] = v4 + 8;
      }
    }
    return result;
  }
  v13 = 0;
  if ( v3 == 54 || v3 == 55 )
  {
    if ( *(_DWORD *)(**(_QWORD **)(a2 - 24) + 8LL) >> 8 )
      return result;
LABEL_11:
    v5 = (_BYTE *)a1[4];
    if ( v5 == (_BYTE *)a1[5] )
      return (__int64)sub_170B610((__int64)(a1 + 3), v5, v11);
    if ( v5 )
    {
      *(_QWORD *)v5 = result;
      v5 = (_BYTE *)a1[4];
    }
    a1[4] = v5 + 8;
    return result;
  }
  if ( v3 != 78 )
  {
    v14[0] = 0;
    if ( (unsigned __int8)(v3 - 58) > 1u )
      return result;
    goto LABEL_11;
  }
  v6 = *(_QWORD *)(a2 - 24);
  if ( !*(_BYTE *)(v6 + 16) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
  {
    v14[0] = a2;
    v7 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v7 + 16) )
      BUG();
    v8 = *(_DWORD *)(v7 + 36);
    v12 = 0;
    result = sub_1C98880((__int64)a1, v8, &v12);
    if ( (_BYTE)result )
    {
      result = **(_QWORD **)(v14[0] + 24 * (v12 - (unsigned __int64)(*(_DWORD *)(v14[0] + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(result + 8) == 15 )
      {
        result = *(_DWORD *)(result + 8) >> 8;
        if ( !(_DWORD)result )
        {
          v10 = (_BYTE *)a1[4];
          if ( v10 == (_BYTE *)a1[5] )
          {
            result = (__int64)sub_170B610((__int64)(a1 + 3), v10, v11);
          }
          else
          {
            if ( v10 )
            {
              result = v11[0];
              *(_QWORD *)v10 = v11[0];
            }
            a1[4] += 8LL;
          }
        }
      }
      if ( (v8 & 0xFFFFFFFD) == 0x85 )
      {
        result = **(_QWORD **)(v14[0] + 24 * (1LL - (*(_DWORD *)(v14[0] + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(result + 8) == 15 )
        {
          result = *(_DWORD *)(result + 8) >> 8;
          if ( !(_DWORD)result )
          {
            v9 = (_BYTE *)a1[7];
            if ( v9 == (_BYTE *)a1[8] )
            {
              return (__int64)sub_1C991F0((__int64)(a1 + 6), v9, v14);
            }
            else
            {
              if ( v9 )
                *(_QWORD *)v9 = v14[0];
              a1[7] += 8LL;
            }
          }
        }
      }
    }
  }
  return result;
}
