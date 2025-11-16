// Function: sub_9B6040
// Address: 0x9b6040
//
__int64 __fastcall sub_9B6040(__int64 a1, __int64 a2, __m128i *a3, unsigned int a4)
{
  __int64 result; // rax
  __int64 v6; // rdx
  unsigned int v8; // eax
  unsigned __int64 v9; // rdx
  unsigned __int8 v10; // [rsp-40h] [rbp-40h]
  __m128i *v11; // [rsp-40h] [rbp-40h]
  __int64 v12; // [rsp-38h] [rbp-38h] BYREF
  unsigned int v13; // [rsp-30h] [rbp-30h]

  if ( a1 == a2 )
    return 0;
  result = 0;
  v6 = *(_QWORD *)(a1 + 8);
  if ( v6 == *(_QWORD *)(a2 + 8) )
  {
    if ( *(_BYTE *)(v6 + 8) == 17 )
    {
      v8 = *(_DWORD *)(v6 + 32);
      v13 = v8;
      if ( v8 > 0x40 )
      {
        v11 = a3;
        sub_C43690(&v12, -1, 1);
        a3 = v11;
      }
      else
      {
        v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
        if ( !v8 )
          v9 = 0;
        v12 = v9;
      }
    }
    else
    {
      v13 = 1;
      v12 = 1;
    }
    result = sub_9B5220((unsigned __int8 *)a1, (unsigned __int8 *)a2, (__int64)&v12, a4, a3);
    if ( v13 > 0x40 )
    {
      if ( v12 )
      {
        v10 = result;
        j_j___libc_free_0_0(v12);
        return v10;
      }
    }
  }
  return result;
}
