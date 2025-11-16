// Function: sub_D5C780
// Address: 0xd5c780
//
__int64 __fastcall sub_D5C780(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // r8
  __int64 result; // rax
  unsigned int v5; // edx
  __int64 v6; // [rsp-40h] [rbp-40h]
  __int64 v7; // [rsp-38h] [rbp-38h] BYREF
  unsigned int v8; // [rsp-30h] [rbp-30h]
  char v9; // [rsp-28h] [rbp-28h]
  __int64 v10; // [rsp-8h] [rbp-8h]

  v3 = *(_QWORD *)(a2 + 8);
  result = a2;
  if ( *(_BYTE *)(v3 + 8) == 12 )
  {
    v10 = v2;
    if ( *(_BYTE *)a2 == 17 )
    {
      v8 = *(_DWORD *)(a2 + 32);
      if ( v8 > 0x40 )
      {
        sub_C43780((__int64)&v7, (const void **)(a2 + 24));
        v3 = *(_QWORD *)(a2 + 8);
      }
      else
      {
        v7 = *(_QWORD *)(a2 + 24);
      }
      v9 = 1;
    }
    else
    {
      v5 = *(unsigned __int8 *)(*(_QWORD *)a1 + 16LL);
      if ( (unsigned __int8)(v5 - 2) > 1u )
        return result;
      sub_D5C200((__int64)&v7, (char *)a2, v5, 0);
      result = a2;
      if ( !v9 )
        return result;
      v3 = *(_QWORD *)(a2 + 8);
    }
    result = sub_AD8D80(v3, (__int64)&v7);
    if ( v9 )
    {
      v9 = 0;
      if ( v8 > 0x40 )
      {
        if ( v7 )
        {
          v6 = result;
          j_j___libc_free_0_0(v7);
          return v6;
        }
      }
    }
  }
  return result;
}
