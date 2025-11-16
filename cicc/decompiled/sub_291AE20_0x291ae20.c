// Function: sub_291AE20
// Address: 0x291ae20
//
unsigned __int64 __fastcall sub_291AE20(__int64 a1, int a2, char a3)
{
  __int64 v3; // rbp
  unsigned __int64 result; // rax
  __int64 v5; // rdx
  __int64 **v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // [rsp-50h] [rbp-50h]
  _WORD v9[32]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v10; // [rsp-8h] [rbp-8h]

  result = *(_QWORD *)(a1 + 32);
  if ( a3 )
  {
    v10 = v3;
    v5 = *(_QWORD *)(result + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
      v5 = **(_QWORD **)(v5 + 16);
    if ( a2 != *(_DWORD *)(v5 + 8) >> 8 )
    {
      v6 = (__int64 **)sub_BCE3C0(*(__int64 **)(a1 + 248), a2);
      v7 = *(_QWORD *)(a1 + 32);
      v9[16] = 257;
      return sub_291AC80((__int64 *)(a1 + 176), 0x32u, v7, v6, (__int64)v9, 0, v8, 0);
    }
  }
  return result;
}
