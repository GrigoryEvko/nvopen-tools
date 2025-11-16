// Function: sub_1B967F0
// Address: 0x1b967f0
//
_QWORD *__fastcall sub_1B967F0(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 *v3; // r14
  __int64 v4; // rdi
  __int64 v5; // r13
  unsigned int v6; // ebx
  unsigned int v7; // eax
  bool v8; // cc
  _QWORD *result; // rax
  _BYTE v10[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v11; // [rsp+10h] [rbp-30h]

  v2 = *a1;
  if ( *(_BYTE *)(a2 + 16) != 61 || (result = *(_QWORD **)(a2 - 24), *v2 != *result) )
  {
    v3 = a1[1];
    v4 = *(_QWORD *)a2;
    v11 = 257;
    v5 = *v2;
    v6 = sub_16431D0(v4);
    v7 = sub_16431D0(v5);
    v8 = v6 <= v7;
    if ( v6 < v7 )
    {
      return (_QWORD *)sub_12AA3B0(v3, 0x25u, a2, v5, (__int64)v10);
    }
    else
    {
      result = (_QWORD *)a2;
      if ( !v8 )
        return (_QWORD *)sub_12AA3B0(v3, 0x24u, a2, v5, (__int64)v10);
    }
  }
  return result;
}
