// Function: sub_39839F0
// Address: 0x39839f0
//
__int64 __fastcall sub_39839F0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // r15
  __int64 (__fastcall *v9)(__int64, _QWORD, __int64, __int64); // r14
  __int64 v10; // rax
  __int64 *i; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(v2 + 240);
  result = *(unsigned int *)(v3 + 348);
  if ( (*(_DWORD *)(v3 + 348) & 0xFFFFFFFD) == 1
    || (_DWORD)result == 4 && (result = *(unsigned int *)(v3 + 352), (_DWORD)result) && (_DWORD)result != 6 )
  {
    result = sub_396DD80(v2);
    v5 = result;
    if ( *(char *)(result + 12) < 0 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(__int64 **)(v6 + 1704);
      result = *(_QWORD *)(v6 + 1712);
      for ( i = (__int64 *)result; i != v7; ++v7 )
      {
        if ( *v7 )
        {
          v8 = sub_396EAF0(*(_QWORD *)(a1 + 8), *v7);
          v9 = *(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v5 + 24LL);
          v10 = sub_396DDB0(*(_QWORD *)(a1 + 8));
          result = v9(v5, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL), v10, v8);
        }
      }
    }
  }
  return result;
}
