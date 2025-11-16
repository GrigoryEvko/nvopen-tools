// Function: sub_31D60D0
// Address: 0x31d60d0
//
__int64 __fastcall sub_31D60D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  unsigned int v4; // ebx
  __int64 *v5; // rax
  __int64 v6; // r14
  void (*v7)(); // r15
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  int v10; // [rsp+Ch] [rbp-34h]

  result = sub_BA8DC0(a2, (__int64)"llvm.ident", 10);
  if ( result )
  {
    v3 = result;
    result = sub_B91A00(result);
    v10 = result;
    if ( (_DWORD)result )
    {
      v4 = 0;
      do
      {
        v8 = sub_B91A10(v3, v4);
        v9 = *(_BYTE *)(v8 - 16);
        if ( (v9 & 2) != 0 )
          v5 = *(__int64 **)(v8 - 32);
        else
          v5 = (__int64 *)(-16 - 8LL * ((v9 >> 2) & 0xF) + v8);
        v6 = *(_QWORD *)(a1 + 224);
        v7 = *(void (**)())(*(_QWORD *)v6 + 648LL);
        result = sub_B91420(*v5);
        if ( v7 != nullsub_107 )
          result = ((__int64 (__fastcall *)(__int64, __int64))v7)(v6, result);
        ++v4;
      }
      while ( v10 != v4 );
    }
  }
  return result;
}
