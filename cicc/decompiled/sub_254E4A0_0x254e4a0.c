// Function: sub_254E4A0
// Address: 0x254e4a0
//
__int64 __fastcall sub_254E4A0(__int64 a1, __int64 (__fastcall *a2)(__int64, unsigned __int64), __int64 a3, char a4)
{
  unsigned int v5; // r14d
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // r13
  __int64 result; // rax
  unsigned __int64 v11; // rax

  v5 = *(unsigned __int8 *)(a1 + 97);
  if ( (_BYTE)v5 )
  {
    v6 = a1 + 104;
    v7 = a1 + 216;
    if ( a4 == 1 )
      v7 = v6;
    v8 = *(unsigned __int64 **)(v7 + 32);
    v9 = &v8[*(unsigned int *)(v7 + 40)];
    if ( v9 == v8 )
    {
      return v5;
    }
    else
    {
      while ( 1 )
      {
        result = a2(a3, *v8);
        if ( !(_BYTE)result )
          break;
        if ( v9 == ++v8 )
          return v5;
      }
    }
  }
  else
  {
    v11 = sub_250D070((_QWORD *)(a1 + 72));
    return a2(a3, v11);
  }
  return result;
}
