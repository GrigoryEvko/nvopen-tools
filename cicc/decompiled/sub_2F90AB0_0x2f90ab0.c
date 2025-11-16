// Function: sub_2F90AB0
// Address: 0x2f90ab0
//
void __fastcall sub_2F90AB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rbx
  __int64 *v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rsi

  if ( *(_BYTE *)(a1 + 16) )
  {
    sub_2F90200(a1, a2, a3, a4, a5, a6);
  }
  else
  {
    v6 = *(__int64 **)(a1 + 24);
    v7 = &v6[2 * *(unsigned int *)(a1 + 32)];
    while ( v7 != v6 )
    {
      v8 = v6[1];
      v9 = *v6;
      v6 += 2;
      sub_2F90A20(a1, v9, v8);
    }
    *(_DWORD *)(a1 + 32) = 0;
  }
}
