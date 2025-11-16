// Function: sub_1300F90
// Address: 0x1300f90
//
__int64 __fastcall sub_1300F90(__int64 a1, int a2, unsigned __int8 a3)
{
  __int64 v5; // r13
  __int64 result; // rax
  unsigned __int32 v7; // edi
  char *v8; // rcx
  __int64 v9; // rsi
  int v10; // edx

  v5 = qword_50579C0[a2];
  result = sub_13177F0(v5, a3);
  if ( a3 )
  {
    *(_QWORD *)(a1 + 136) = v5;
  }
  else
  {
    *(_QWORD *)(a1 + 144) = v5;
    v7 = _InterlockedExchangeAdd((volatile signed __int32 *)(v5 + 8), 1u);
    v8 = (char *)&unk_5260DF4;
    v9 = a1 + 161;
    do
    {
      ++v9;
      v10 = v7 % *(_DWORD *)v8;
      result = v7 / *(_DWORD *)v8;
      v8 += 40;
      *(_BYTE *)(v9 - 1) = v10;
    }
    while ( v8 != (char *)&unk_5260DE0 + 1460 );
  }
  return result;
}
