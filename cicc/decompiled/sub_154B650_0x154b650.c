// Function: sub_154B650
// Address: 0x154b650
//
_BYTE *__fastcall sub_154B650(__int64 a1, const char *a2, size_t a3)
{
  int v4; // edi
  char v5; // r15
  const char *v6; // rbx
  const char *v7; // r13
  void *v8; // rdi
  _BYTE *result; // rax
  _BYTE *v10; // rax

  v4 = *(unsigned __int8 *)a2;
  if ( (unsigned int)(v4 - 48) > 9 )
  {
    if ( !(_DWORD)a3 )
    {
LABEL_9:
      v8 = *(void **)(a1 + 24);
      result = (_BYTE *)(*(_QWORD *)(a1 + 16) - (_QWORD)v8);
      if ( a3 > (unsigned __int64)result )
        return (_BYTE *)sub_16E7EE0(a1, a2, a3);
      if ( a3 )
      {
        result = memcpy(v8, a2, a3);
        *(_QWORD *)(a1 + 24) += a3;
      }
      return result;
    }
    v5 = *a2;
    v6 = a2;
    v7 = &a2[(unsigned int)(a3 - 1)];
    while ( isalnum(v4) || (unsigned __int8)(v5 - 45) <= 1u || v5 == 95 )
    {
      if ( v6 == v7 )
        goto LABEL_9;
      v4 = *(unsigned __int8 *)++v6;
      v5 = v4;
    }
  }
  v10 = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(a1 + 16) )
  {
    sub_16E7DE0(a1, 34);
  }
  else
  {
    *(_QWORD *)(a1 + 24) = v10 + 1;
    *v10 = 34;
  }
  sub_16D16F0(a2, a3, a1);
  result = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(a1 + 16) )
    return (_BYTE *)sub_16E7DE0(a1, 34);
  *(_QWORD *)(a1 + 24) = result + 1;
  *result = 34;
  return result;
}
