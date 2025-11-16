// Function: sub_277FA60
// Address: 0x277fa60
//
__int64 __fastcall sub_277FA60(__int64 a1, unsigned __int8 **a2, __int64 **a3)
{
  int v4; // ebx
  __int64 result; // rax
  __int64 v6; // r13
  int v7; // ebx
  __int64 *v8; // r14
  __int64 *v9; // [rsp+0h] [rbp-40h]
  int v10; // [rsp+8h] [rbp-38h]
  unsigned int v11; // [rsp+Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v4 - 1;
    v10 = 1;
    v9 = 0;
    v11 = v7 & sub_277F590(*a2);
    while ( 1 )
    {
      v8 = (__int64 *)(v6 + 16LL * v11);
      result = sub_277AC50((__int64)*a2, *v8);
      if ( (_BYTE)result )
        break;
      if ( *v8 == -4096 )
      {
        if ( v9 )
          v8 = v9;
        break;
      }
      if ( *v8 == -8192 )
      {
        if ( v9 )
          v8 = v9;
        v9 = v8;
      }
      v11 = v7 & (v10 + v11);
      ++v10;
    }
    *a3 = v8;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return result;
}
