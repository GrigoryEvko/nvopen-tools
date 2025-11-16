// Function: sub_27E9540
// Address: 0x27e9540
//
__int64 __fastcall sub_27E9540(__int64 a1, __int64 a2, _QWORD *a3, unsigned int a4, unsigned __int8 *a5)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  __int64 *v10; // rax
  unsigned __int8 *v11; // [rsp+8h] [rbp-28h]

  if ( *(_BYTE *)(a1 + 124) )
  {
    v7 = *(_QWORD **)(a1 + 104);
    v8 = &v7[*(unsigned int *)(a1 + 116)];
    if ( v7 == v8 )
      return sub_27E8360(a1, a2, a3, a4, a5);
    while ( a3 != (_QWORD *)*v7 )
    {
      if ( v8 == ++v7 )
        return sub_27E8360(a1, a2, a3, a4, a5);
    }
    return 0;
  }
  v11 = a5;
  v10 = sub_C8CA60(a1 + 96, (__int64)a3);
  a5 = v11;
  if ( v10 )
    return 0;
  return sub_27E8360(a1, a2, a3, a4, a5);
}
