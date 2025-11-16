// Function: sub_2980F90
// Address: 0x2980f90
//
__int64 __fastcall sub_2980F90(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6)
{
  __int64 v11; // rdi
  _QWORD *v12; // rax
  unsigned int v13; // edx
  __int64 v14; // rsi
  __int64 v15; // rax

  v11 = sub_AE4570(*a1, *(_QWORD *)(a6 + 8));
  v12 = *(_QWORD **)(a3 + 24);
  v13 = *(_DWORD *)(a3 + 32);
  if ( v13 > 0x40 )
  {
    v14 = a5 * *v12;
  }
  else
  {
    v14 = 0;
    if ( v13 )
      v14 = a5 * ((__int64)((_QWORD)v12 << (64 - (unsigned __int8)v13)) >> (64 - (unsigned __int8)v13));
  }
  v15 = sub_ACD640(v11, v14, 1u);
  return sub_297F050((__int64)a1, 3, a2, v15, a4, a6);
}
