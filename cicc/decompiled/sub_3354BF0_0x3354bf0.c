// Function: sub_3354BF0
// Address: 0x3354bf0
//
__int64 __fastcall sub_3354BF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        void (__fastcall *a4)(__int64 a1, __int64 a2),
        __int64 a5,
        __int64 a6)
{
  __int64 v7; // rax
  unsigned int *v8; // rdi
  __int64 result; // rax
  void (*v10)(void); // rax

  v7 = *(_QWORD *)(a1 + 640);
  *(_DWORD *)(a1 + 688) = 0;
  *(_DWORD *)(v7 + 8) = a2;
  v8 = *(unsigned int **)(a1 + 672);
  result = v8[2];
  if ( (_DWORD)result )
  {
    if ( (_DWORD)a2 != *(_DWORD *)(a1 + 680) )
    {
      while ( 1 )
      {
        v10 = *(void (**)(void))(*(_QWORD *)v8 + 88LL);
        if ( v10 != nullsub_1621 )
          v10();
        result = (unsigned int)(*(_DWORD *)(a1 + 680) + 1);
        *(_DWORD *)(a1 + 680) = result;
        if ( (_DWORD)a2 == (_DWORD)result )
          break;
        v8 = *(unsigned int **)(a1 + 672);
      }
    }
  }
  else
  {
    *(_DWORD *)(a1 + 680) = a2;
  }
  if ( !byte_5038F08 )
    return sub_33549F0(a1, a2, a3, a4, a5, a6);
  return result;
}
