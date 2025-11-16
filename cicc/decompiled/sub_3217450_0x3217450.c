// Function: sub_3217450
// Address: 0x3217450
//
__int64 __fastcall sub_3217450(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // r13
  __int64 *v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r15
  void (__fastcall *v11)(__int64, _QWORD, __int64, __int64, __int64); // r14
  __int64 v12; // rax
  __int64 *v13; // [rsp+0h] [rbp-40h]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = a1[1];
  v3 = *(_QWORD *)(v2 + 208);
  result = *(unsigned int *)(v3 + 336);
  if ( (*(_DWORD *)(v3 + 336) & 0xFFFFFFFD) == 1
    || (_DWORD)result == 7
    || (_DWORD)result == 4 && (result = *(unsigned int *)(v3 + 344), (_DWORD)result) && (_DWORD)result != 6 )
  {
    result = sub_31DA6B0(v2);
    v5 = result;
    if ( *(char *)(result + 940) < 0 )
    {
      result = a1[5];
      v6 = (__int64 *)a1[4];
      v13 = (__int64 *)result;
      if ( v6 != (__int64 *)result )
      {
        do
        {
          v7 = *v6++;
          v8 = sub_31DB510(a1[1], v7);
          v9 = a1[1];
          v10 = v8;
          v11 = *(void (__fastcall **)(__int64, _QWORD, __int64, __int64, __int64))(*(_QWORD *)v5 + 32LL);
          v14 = *(_QWORD *)(v9 + 240);
          v12 = sub_31DA930(v9);
          v11(v5, *(_QWORD *)(a1[1] + 224), v12, v10, v14);
        }
        while ( v13 != v6 );
        result = a1[4];
        if ( result != a1[5] )
          a1[5] = result;
      }
    }
  }
  return result;
}
