// Function: sub_128F980
// Address: 0x128f980
//
char *__fastcall sub_128F980(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 *v6[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( !a2 || sub_127B420(*(_QWORD *)a2) )
    sub_127B550("unexpected non-scalar type expression!", (_DWORD *)(a2 + 36), 1);
  v6[0] = (__int64 *)a1;
  v6[1] = (__int64 *)(a1 + 48);
  v6[2] = *(__int64 **)(a1 + 40);
  return sub_128D0F0(v6, a2, v2, v3, v4);
}
