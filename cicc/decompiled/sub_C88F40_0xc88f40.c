// Function: sub_C88F40
// Address: 0xc88f40
//
__int64 __fastcall sub_C88F40(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  int v7; // edx
  __int64 result; // rax

  v6 = sub_22077B0(32);
  if ( v6 )
  {
    *(_DWORD *)v6 = 0;
    *(_QWORD *)(v6 + 8) = 0;
    *(_QWORD *)(v6 + 24) = 0;
  }
  *(_QWORD *)a1 = v6;
  v7 = a4 & 1;
  *(_QWORD *)(v6 + 16) = a2 + a3;
  if ( (a4 & 1) != 0 )
    v7 = 2;
  if ( (a4 & 2) != 0 )
    v7 |= 8u;
  if ( (a4 & 4) == 0 )
    v7 |= 1u;
  result = sub_CBA570(v6, a2, v7 | 0x20u);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
