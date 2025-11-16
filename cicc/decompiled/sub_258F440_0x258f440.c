// Function: sub_258F440
// Address: 0x258f440
//
__int64 __fastcall sub_258F440(__int64 a1, _QWORD *a2, __int64 *a3, __int64 a4)
{
  char v6; // al
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  char v13; // [rsp+Fh] [rbp-29h] BYREF
  __int64 v14[5]; // [rsp+10h] [rbp-28h] BYREF

  v6 = sub_258F340(a2, a1, (__m128i *)(a1 + 72), 2, &v13, 0, 0);
  v7 = *(unsigned int *)(a1 + 108);
  if ( v6 )
    v14[0] = sub_A77A80(a3, v7);
  else
    v14[0] = sub_A77A90(a3, v7);
  return sub_25594F0(a4, v14, v8, v9, v10, v11);
}
