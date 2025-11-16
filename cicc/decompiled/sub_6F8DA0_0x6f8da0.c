// Function: sub_6F8DA0
// Address: 0x6f8da0
//
__int64 __fastcall sub_6F8DA0(__int64 *a1, _QWORD *a2, int a3, unsigned int a4, _QWORD *a5, _DWORD *a6)
{
  __int64 v10; // rbx
  __int64 v11; // r9
  __int64 *v12; // rdi
  __int64 result; // rax
  __int64 v15; // [rsp+18h] [rbp-1B8h]
  _BYTE v16[432]; // [rsp+20h] [rbp-1B0h] BYREF

  v15 = *a1;
  v10 = sub_6E3DA0(*a1, (__int64)v16);
  v12 = *(__int64 **)(v15 + 72);
  if ( a3 && *(_BYTE *)(v15 + 24) == 1 && (unsigned __int8)(*(_BYTE *)(v15 + 56) - 105) <= 4u )
    v12 = (__int64 *)v12[2];
  if ( a4 )
    a4 = 128;
  sub_6F85E0(v12, (__int64)a1, a4, a2, 0, v11);
  *a5 = *(_QWORD *)(v10 + 356);
  result = *(unsigned int *)(v10 + 364);
  *a6 = result;
  return result;
}
