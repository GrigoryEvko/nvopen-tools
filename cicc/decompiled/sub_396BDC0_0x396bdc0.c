// Function: sub_396BDC0
// Address: 0x396bdc0
//
__int64 __fastcall sub_396BDC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  unsigned int v4; // r15d
  __int64 v5; // rax
  __int64 v6; // r13
  void (*v7)(); // r14
  int v8; // [rsp+Ch] [rbp-54h]
  const char *v9; // [rsp+10h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-40h]
  char v11; // [rsp+21h] [rbp-3Fh]

  v11 = 1;
  v9 = "llvm.ident";
  v10 = 3;
  result = sub_1632310(a2, (__int64)&v9);
  if ( result )
  {
    v3 = result;
    result = sub_161F520(result);
    v8 = result;
    if ( (_DWORD)result )
    {
      v4 = 0;
      do
      {
        while ( 1 )
        {
          v5 = sub_161F530(v3, v4);
          v6 = *(_QWORD *)(a1 + 256);
          v7 = *(void (**)())(*(_QWORD *)v6 + 560LL);
          result = sub_161E970(*(_QWORD *)(v5 - 8LL * *(unsigned int *)(v5 + 8)));
          if ( v7 != nullsub_589 )
            break;
          if ( v8 == ++v4 )
            return result;
        }
        ++v4;
        result = ((__int64 (__fastcall *)(__int64, __int64))v7)(v6, result);
      }
      while ( v8 != v4 );
    }
  }
  return result;
}
