// Function: sub_1D04A20
// Address: 0x1d04a20
//
__int64 __fastcall sub_1D04A20(__int64 a1, int a2)
{
  __int64 v3; // rax
  unsigned int *v4; // rdi
  __int64 result; // rax
  void (*v6)(void); // rax

  v3 = *(_QWORD *)(a1 + 672);
  *(_DWORD *)(a1 + 720) = 0;
  *(_DWORD *)(v3 + 8) = a2;
  v4 = *(unsigned int **)(a1 + 704);
  result = v4[2];
  if ( (_DWORD)result )
  {
    if ( a2 != *(_DWORD *)(a1 + 712) )
    {
      while ( 1 )
      {
        v6 = *(void (**)(void))(*(_QWORD *)v4 + 88LL);
        if ( v6 != nullsub_680 )
          v6();
        result = (unsigned int)(*(_DWORD *)(a1 + 712) + 1);
        *(_DWORD *)(a1 + 712) = result;
        if ( a2 == (_DWORD)result )
          break;
        v4 = *(unsigned int **)(a1 + 704);
      }
    }
  }
  else
  {
    *(_DWORD *)(a1 + 712) = a2;
  }
  if ( !byte_4FC13A0 )
    return sub_1D04820(a1);
  return result;
}
