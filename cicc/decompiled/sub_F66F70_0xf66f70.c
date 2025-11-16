// Function: sub_F66F70
// Address: 0xf66f70
//
__int64 __fastcall sub_F66F70(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rdi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 *v10; // rbx
  __int64 *v11; // r14
  __int64 v12; // r13
  _QWORD *v13; // [rsp+0h] [rbp-240h] BYREF
  __int64 v14; // [rsp+8h] [rbp-238h]
  _QWORD v15[70]; // [rsp+10h] [rbp-230h] BYREF

  v13 = v15;
  v15[0] = a1;
  v6 = v15;
  v14 = 0x4000000001LL;
  LODWORD(result) = 1;
  do
  {
    v8 = (unsigned int)result;
    result = (unsigned int)(result - 1);
    v9 = v6[v8 - 1];
    LODWORD(v14) = result;
    v10 = *(__int64 **)(v9 + 24);
    *(_DWORD *)(v9 + 16) = *(_DWORD *)(*(_QWORD *)(v9 + 8) + 16LL) + 1;
    v11 = &v10[*(unsigned int *)(v9 + 32)];
    if ( v10 != v11 )
    {
      do
      {
        v12 = *v10;
        if ( *(_DWORD *)(*v10 + 16) != *(_DWORD *)(*(_QWORD *)(*v10 + 8) + 16LL) + 1 )
        {
          if ( result + 1 > (unsigned __int64)HIDWORD(v14) )
          {
            a2 = v15;
            sub_C8D5F0((__int64)&v13, v15, result + 1, 8u, a5, a6);
            result = (unsigned int)v14;
          }
          v13[result] = v12;
          result = (unsigned int)(v14 + 1);
          LODWORD(v14) = v14 + 1;
        }
        ++v10;
      }
      while ( v11 != v10 );
      v6 = v13;
    }
  }
  while ( (_DWORD)result );
  if ( v6 != v15 )
    return _libc_free(v6, a2);
  return result;
}
