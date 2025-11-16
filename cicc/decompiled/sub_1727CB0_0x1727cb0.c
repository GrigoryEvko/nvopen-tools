// Function: sub_1727CB0
// Address: 0x1727cb0
//
unsigned __int8 *__fastcall sub_1727CB0(char a1, int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r12
  bool v10; // cc
  __int64 v11; // rax
  int v12; // [rsp+Ch] [rbp-44h] BYREF
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  __int16 v14; // [rsp+20h] [rbp-30h]

  v8 = sub_14CF630(a1, a2, (__int64 **)a3, a4, &v12);
  if ( !v8 )
  {
    v10 = *(_BYTE *)(a3 + 16) <= 0x10u;
    v14 = 257;
    if ( v10 && *(_BYTE *)(a4 + 16) <= 0x10u )
    {
      v8 = sub_15A37B0(v12, (_QWORD *)a3, (_QWORD *)a4, 0);
      v11 = sub_14DBA30(v8, *(_QWORD *)(a5 + 96), 0);
      if ( v11 )
        return (unsigned __int8 *)v11;
    }
    else
    {
      return sub_1727440(a5, v12, a3, a4, &v13);
    }
  }
  return (unsigned __int8 *)v8;
}
