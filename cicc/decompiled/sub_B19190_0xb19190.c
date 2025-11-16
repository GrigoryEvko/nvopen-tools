// Function: sub_B19190
// Address: 0xb19190
//
__int64 __fastcall sub_B19190(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  _QWORD *v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 *v6; // rbx
  __int64 *v7; // r14
  __int64 v8; // r13
  _QWORD *v9; // [rsp-248h] [rbp-248h] BYREF
  __int64 v10; // [rsp-240h] [rbp-240h]
  _QWORD v11[71]; // [rsp-238h] [rbp-238h] BYREF

  result = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL) + 1);
  if ( *(_DWORD *)(a1 + 16) != (_DWORD)result )
  {
    v9 = v11;
    v11[0] = a1;
    v3 = v11;
    v10 = 0x4000000001LL;
    LODWORD(result) = 1;
    do
    {
      v4 = (unsigned int)result;
      result = (unsigned int)(result - 1);
      v5 = v3[v4 - 1];
      LODWORD(v10) = result;
      v6 = *(__int64 **)(v5 + 24);
      *(_DWORD *)(v5 + 16) = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 16LL) + 1;
      v7 = &v6[*(unsigned int *)(v5 + 32)];
      if ( v7 != v6 )
      {
        do
        {
          v8 = *v6;
          if ( *(_DWORD *)(*v6 + 16) != *(_DWORD *)(*(_QWORD *)(*v6 + 8) + 16LL) + 1 )
          {
            if ( result + 1 > (unsigned __int64)HIDWORD(v10) )
            {
              a2 = v11;
              sub_C8D5F0(&v9, v11, result + 1, 8);
              result = (unsigned int)v10;
            }
            v9[result] = v8;
            result = (unsigned int)(v10 + 1);
            LODWORD(v10) = v10 + 1;
          }
          ++v6;
        }
        while ( v7 != v6 );
        v3 = v9;
      }
    }
    while ( (_DWORD)result );
    if ( v3 != v11 )
      return _libc_free(v3, a2);
  }
  return result;
}
