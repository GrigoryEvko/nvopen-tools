// Function: sub_8E9250
// Address: 0x8e9250
//
__int64 __fastcall sub_8E9250(unsigned __int8 *a1, unsigned int a2, __int64 a3)
{
  unsigned __int8 *v4; // r12
  unsigned __int8 v5; // al
  __int64 result; // rax
  unsigned __int8 v8; // al
  unsigned __int8 *v9; // r12
  unsigned __int8 *v10; // rax
  __int64 v11; // [rsp+8h] [rbp-48h]
  _BYTE v12[64]; // [rsp+10h] [rbp-40h] BYREF

  v4 = a1;
  v5 = *a1;
  if ( *a1 != 84 )
    goto LABEL_2;
  while ( 1 )
  {
    v8 = v4[1];
    switch ( v8 )
    {
      case 'V':
        if ( !*(_QWORD *)(a3 + 32) )
          sub_8E5790("Virtual function table for ", a3);
        goto LABEL_24;
      case 'T':
        if ( !*(_QWORD *)(a3 + 32) )
          sub_8E5790("Virtual table table for ", a3);
        goto LABEL_24;
      case 'I':
        if ( !*(_QWORD *)(a3 + 32) )
          sub_8E5790("Typeinfo for ", a3);
        goto LABEL_24;
      case 'S':
        if ( !*(_QWORD *)(a3 + 32) )
          sub_8E5790("Typeinfo name for ", a3);
LABEL_24:
        v9 = v4 + 2;
        v11 = sub_8E9FF0(v9, 0, 0, 0, 1, a3);
        sub_8EB260(v9, 0, 0, a3);
        return v11;
      case 'c':
        if ( !*(_QWORD *)(a3 + 32) )
          sub_8E5790("Covariant thunk for ", a3);
        v10 = sub_8E5EA0(v4 + 2, a3);
        v4 = sub_8E5EA0(v10, a3);
        goto LABEL_34;
    }
    if ( v8 != 118 && v8 != 104 )
      break;
    if ( !*(_QWORD *)(a3 + 32) )
      sub_8E5790("Thunk for ", a3);
    v4 = sub_8E5EA0(v4 + 1, a3);
LABEL_34:
    v5 = *v4;
    if ( *v4 != 84 )
    {
      a2 = 1;
LABEL_2:
      if ( v5 != 71 || v4[1] != 86 )
      {
        sub_8EDAE0(v4, a2, 1, a3);
        result = (__int64)v4;
        if ( !*(_DWORD *)(a3 + 24) )
          return sub_8EDAE0(v4, a2, 0, a3);
        return result;
      }
      if ( !*(_QWORD *)(a3 + 32) )
        sub_8E5790("Initialization guard variable for ", a3);
      return sub_8E9510(v4 + 2, v12, 3, a3);
    }
  }
  switch ( v8 )
  {
    case 'H':
      if ( !*(_QWORD *)(a3 + 32) )
        sub_8E5790("Thread-local initialization routine for ", a3);
      return sub_8E9510(v4 + 2, v12, 3, a3);
    case 'W':
      if ( !*(_QWORD *)(a3 + 32) )
        sub_8E5790("Thread-local wrapper routine for ", a3);
      return sub_8E9510(v4 + 2, v12, 3, a3);
    case 'A':
      if ( !*(_QWORD *)(a3 + 32) )
        sub_8E5790("template parameter object for ", a3);
      return sub_8E8F40(v4 + 2, a3);
    default:
      result = (__int64)v4;
      if ( !*(_DWORD *)(a3 + 24) )
      {
        ++*(_QWORD *)(a3 + 32);
        ++*(_QWORD *)(a3 + 48);
        *(_DWORD *)(a3 + 24) = 1;
      }
      break;
  }
  return result;
}
