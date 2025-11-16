// Function: sub_98A0C0
// Address: 0x98a0c0
//
_QWORD *__fastcall sub_98A0C0(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v8; // r8
  _BYTE *v10; // rax
  __int64 v11; // [rsp+8h] [rbp-28h]

  if ( *(_BYTE *)a5 == 18 )
  {
    v8 = a5 + 24;
LABEL_3:
    sub_989E10((__int64)a1, a2, a3, a4, v8, a6);
    return a1;
  }
  v11 = a4;
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a5 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)a5 <= 0x15u )
  {
    v10 = (_BYTE *)sub_AD7630(a5, 1);
    if ( v10 )
    {
      if ( *v10 == 18 )
      {
        a4 = v11;
        v8 = (__int64)(v10 + 24);
        goto LABEL_3;
      }
    }
  }
  a1[1] = 0;
  *a1 = 0x3FF000003FFLL;
  return a1;
}
