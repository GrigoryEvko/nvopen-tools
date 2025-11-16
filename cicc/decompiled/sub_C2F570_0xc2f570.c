// Function: sub_C2F570
// Address: 0xc2f570
//
__int64 __fastcall sub_C2F570(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 result; // rax
  _QWORD v5[3]; // [rsp+8h] [rbp-18h] BYREF

  sub_C2F370(*(_QWORD *)(a1 + 8));
  v2 = *(_QWORD *)(a1 + 8);
  v5[0] = 0;
  sub_CB6200(v2, v5, 8);
  v3 = *(_QWORD *)(a1 + 8);
  v5[0] = 0;
  result = sub_CB6200(v3, v5, 8);
  if ( *(_BYTE *)(a1 + 32) )
    return sub_C2EF90(*(_QWORD *)(a1 + 8), *(const void **)(a1 + 16), *(_QWORD *)(a1 + 24));
  return result;
}
