// Function: sub_16F48F0
// Address: 0x16f48f0
//
__int64 __fastcall sub_16F48F0(__int64 a1)
{
  unsigned __int8 *v1; // rdx
  __int64 result; // rax
  char v3; // r8
  char v4; // cl
  __int64 v5; // rcx
  unsigned __int8 v6; // si
  char v7; // si

  v1 = (unsigned __int8 *)(*(_QWORD *)(a1 + 16) + *(_QWORD *)(a1 + 24));
  result = *v1;
  v3 = *(_BYTE *)(a1 + 9);
  if ( (_BYTE)result == 10 )
  {
    ++v1;
    goto LABEL_49;
  }
  if ( (_BYTE)result != 13 )
  {
    if ( !v3 )
      goto LABEL_4;
    goto LABEL_11;
  }
  if ( v1[1] == 10 )
  {
    v1 += 2;
LABEL_49:
    ++*(_DWORD *)(a1 + 12);
    result = *v1;
    if ( !v3 )
    {
      v5 = 0;
      if ( (_BYTE)result == 10 )
        goto LABEL_20;
      if ( (_BYTE)result != 13 )
      {
LABEL_4:
        v4 = *(_BYTE *)(a1 + 8);
        if ( v4 )
          goto LABEL_5;
LABEL_8:
        if ( !(_BYTE)result )
        {
LABEL_9:
          *(_QWORD *)a1 = 0;
          *(_QWORD *)(a1 + 16) = 0;
          *(_QWORD *)(a1 + 24) = 0;
          return result;
        }
        goto LABEL_16;
      }
      if ( v1[1] == 10 )
      {
LABEL_15:
        result = 13;
LABEL_16:
        v5 = 0;
        while ( (_BYTE)result != 13 || v1[v5 + 1] != 10 )
        {
          result = v1[++v5];
          if ( (_BYTE)result == 10 || !(_BYTE)result )
            goto LABEL_20;
        }
        *(_QWORD *)(a1 + 16) = v1;
        *(_QWORD *)(a1 + 24) = v5;
        return result;
      }
    }
  }
LABEL_11:
  v4 = *(_BYTE *)(a1 + 8);
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
          goto LABEL_15;
        v1 += 2;
      }
      ++*(_DWORD *)(a1 + 12);
      result = *v1;
    }
  }
  while ( 1 )
  {
LABEL_5:
    if ( (_BYTE)result == 10 )
      goto LABEL_26;
    if ( (_BYTE)result == 13 )
      break;
    if ( (_BYTE)result != v4 )
      goto LABEL_8;
LABEL_32:
    result = *++v1;
    if ( (_BYTE)result == 10 )
    {
LABEL_37:
      ++v1;
      goto LABEL_38;
    }
    if ( (_BYTE)result )
    {
LABEL_34:
      while ( 1 )
      {
        v7 = result;
        result = v1[1];
        if ( v7 == 13 && (_BYTE)result == 10 )
          break;
        ++v1;
        if ( !(_BYTE)result )
          goto LABEL_28;
        if ( (_BYTE)result == 10 )
          goto LABEL_37;
      }
      result = *v1;
    }
LABEL_28:
    if ( (_BYTE)result == 10 )
      goto LABEL_37;
    if ( (_BYTE)result != 13 )
      goto LABEL_8;
    if ( v1[1] != 10 )
      goto LABEL_23;
    v1 += 2;
LABEL_38:
    ++*(_DWORD *)(a1 + 12);
    result = *v1;
  }
  v6 = v1[1];
  if ( v6 == 10 )
  {
LABEL_26:
    if ( !v3 )
      goto LABEL_24;
    if ( (_BYTE)result != v4 )
      goto LABEL_28;
    goto LABEL_32;
  }
  if ( v4 == 13 )
  {
    ++v1;
    if ( v6 )
    {
      LOBYTE(result) = v6;
      goto LABEL_34;
    }
    goto LABEL_9;
  }
LABEL_23:
  result = *v1;
LABEL_24:
  if ( (_BYTE)result != 10 )
    goto LABEL_16;
  v5 = 0;
LABEL_20:
  *(_QWORD *)(a1 + 16) = v1;
  *(_QWORD *)(a1 + 24) = v5;
  return result;
}
