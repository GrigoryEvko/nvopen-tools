// Function: sub_92F410
// Address: 0x92f410
//
__m128i *__fastcall sub_92F410(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v4[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( !a2 || sub_91B770(*(_QWORD *)a2) )
    sub_91B8A0("unexpected non-scalar type expression!", (_DWORD *)(a2 + 36), 1);
  v4[0] = a1;
  v4[1] = a1 + 48;
  v4[2] = *(_QWORD *)(a1 + 40);
  return sub_92CBF0(v4, a2, v2);
}
