// Function: sub_C7C5C0
// Address: 0xc7c5c0
//
__int64 __fastcall sub_C7C5C0(__int64 a1)
{
  unsigned __int8 *v1; // rdx
  __int64 result; // rax
  char v3; // r8
  char v4; // cl
  __int64 v5; // rcx
  unsigned __int8 v6; // si
  char v7; // si

  v1 = (unsigned __int8 *)(*(_QWORD *)(a1 + 48) + *(_QWORD *)(a1 + 56));
  result = *v1;
  v3 = *(_BYTE *)(a1 + 41);
  if ( (_BYTE)result == 10 )
  {
    ++v1;
    goto LABEL_51;
  }
  if ( (_BYTE)result != 13 )
  {
    if ( !v3 )
      goto LABEL_4;
    goto LABEL_13;
  }
  if ( v1[1] == 10 )
  {
    v1 += 2;
LABEL_51:
    ++*(_DWORD *)(a1 + 44);
    result = *v1;
    if ( !v3 )
    {
      v5 = 0;
      if ( (_BYTE)result == 10 )
        goto LABEL_22;
      if ( (_BYTE)result != 13 )
      {
LABEL_4:
        v4 = *(_BYTE *)(a1 + 40);
        if ( v4 )
          goto LABEL_5;
LABEL_8:
        if ( !(_BYTE)result )
        {
LABEL_9:
          if ( *(_BYTE *)(a1 + 32) )
            *(_BYTE *)(a1 + 32) = 0;
          *(_QWORD *)(a1 + 48) = 0;
          *(_QWORD *)(a1 + 56) = 0;
          return result;
        }
        goto LABEL_18;
      }
      if ( v1[1] == 10 )
      {
LABEL_17:
        result = 13;
LABEL_18:
        v5 = 0;
        while ( (_BYTE)result != 13 || v1[v5 + 1] != 10 )
        {
          result = v1[++v5];
          if ( (_BYTE)result == 10 || !(_BYTE)result )
            goto LABEL_22;
        }
        *(_QWORD *)(a1 + 48) = v1;
        *(_QWORD *)(a1 + 56) = v5;
        return result;
      }
    }
  }
LABEL_13:
  v4 = *(_BYTE *)(a1 + 40);
  if ( !v4 )
  {
    while ( 1 )
    {
      if ( (_BYTE)result == 10 )
      {
        ++v1;
      }
      else
      {
        if ( (_BYTE)result != 13 )
          goto LABEL_8;
        if ( v1[1] != 10 )
          goto LABEL_17;
        v1 += 2;
      }
      ++*(_DWORD *)(a1 + 44);
      result = *v1;
    }
  }
  while ( 1 )
  {
LABEL_5:
    if ( (_BYTE)result == 10 )
      goto LABEL_28;
    if ( (_BYTE)result == 13 )
      break;
    if ( (_BYTE)result != v4 )
      goto LABEL_8;
LABEL_34:
    result = *++v1;
    if ( (_BYTE)result == 10 )
    {
LABEL_39:
      ++v1;
      goto LABEL_40;
    }
    if ( (_BYTE)result )
    {
LABEL_36:
      while ( 1 )
      {
        v7 = result;
        result = v1[1];
        if ( v7 == 13 && (_BYTE)result == 10 )
          break;
        ++v1;
        if ( !(_BYTE)result )
          goto LABEL_30;
        if ( (_BYTE)result == 10 )
          goto LABEL_39;
      }
      result = *v1;
    }
LABEL_30:
    if ( (_BYTE)result == 10 )
      goto LABEL_39;
    if ( (_BYTE)result != 13 )
      goto LABEL_8;
    if ( v1[1] != 10 )
      goto LABEL_25;
    v1 += 2;
LABEL_40:
    ++*(_DWORD *)(a1 + 44);
    result = *v1;
  }
  v6 = v1[1];
  if ( v6 == 10 )
  {
LABEL_28:
    if ( !v3 )
      goto LABEL_26;
    if ( (_BYTE)result != v4 )
      goto LABEL_30;
    goto LABEL_34;
  }
  if ( v4 == 13 )
  {
    ++v1;
    if ( v6 )
    {
      LOBYTE(result) = v6;
      goto LABEL_36;
    }
    goto LABEL_9;
  }
LABEL_25:
  result = *v1;
LABEL_26:
  if ( (_BYTE)result != 10 )
    goto LABEL_18;
  v5 = 0;
LABEL_22:
  *(_QWORD *)(a1 + 48) = v1;
  *(_QWORD *)(a1 + 56) = v5;
  return result;
}
