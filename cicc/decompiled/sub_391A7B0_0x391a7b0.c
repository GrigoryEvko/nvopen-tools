// Function: sub_391A7B0
// Address: 0x391a7b0
//
_BYTE *__fastcall sub_391A7B0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int i; // r13d
  char v6; // r12
  _BYTE *result; // rax
  char v8; // si
  unsigned int v9; // r15d
  char v10; // al
  unsigned int v11; // ecx
  __int64 v12; // r12
  char v13; // bl
  char *v14; // rax
  unsigned int v15; // [rsp+Ch] [rbp-34h]
  unsigned int v16; // [rsp+Ch] [rbp-34h]

  for ( i = 1; ; i = v9 )
  {
    v10 = a1;
    v8 = a1 & 0x7F;
    a1 >>= 7;
    if ( a1 )
    {
      if ( a1 != -1 || (v10 & 0x40) == 0 )
      {
        v6 = 1;
        goto LABEL_3;
      }
    }
    else
    {
      v6 = 1;
      if ( (v10 & 0x40) != 0 )
        goto LABEL_3;
    }
    v6 = 0;
    if ( a3 <= i )
      break;
LABEL_3:
    result = *(_BYTE **)(a2 + 24);
    v8 |= 0x80u;
    if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 16) )
      goto LABEL_12;
LABEL_4:
    v9 = i + 1;
    *(_QWORD *)(a2 + 24) = result + 1;
    *result = v8;
    if ( !v6 )
      goto LABEL_13;
LABEL_5:
    ;
  }
  result = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)result < *(_QWORD *)(a2 + 16) )
    goto LABEL_4;
LABEL_12:
  v15 = a3;
  v9 = i + 1;
  result = (_BYTE *)sub_16E7DE0(a2, v8);
  a3 = v15;
  if ( v6 )
    goto LABEL_5;
LABEL_13:
  if ( a3 > i )
  {
    v11 = a3 - 1;
    v12 = a1 >> 63;
    v13 = (a1 >> 63) | 0x80;
    if ( i < a3 - 1 )
    {
      while ( 1 )
      {
        v14 = *(char **)(a2 + 24);
        if ( (unsigned __int64)v14 < *(_QWORD *)(a2 + 16) )
        {
          *(_QWORD *)(a2 + 24) = v14 + 1;
          *v14 = v13;
        }
        else
        {
          v16 = v11;
          sub_16E7DE0(a2, v13);
          v11 = v16;
        }
        if ( v9 == v11 )
          break;
        ++v9;
      }
    }
    result = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 16) )
    {
      return (_BYTE *)sub_16E7DE0(a2, v12 & 0x7F);
    }
    else
    {
      *(_QWORD *)(a2 + 24) = result + 1;
      *result = v12 & 0x7F;
    }
  }
  return result;
}
