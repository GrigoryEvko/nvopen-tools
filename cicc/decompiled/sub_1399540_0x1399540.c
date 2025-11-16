// Function: sub_1399540
// Address: 0x1399540
//
__int64 __fastcall sub_1399540(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13

  v2 = sub_22077B0(72);
  v3 = v2;
  if ( v2 )
    sub_1399480(v2, a2);
  v4 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v3;
  if ( v4 )
  {
    sub_1397630(v4);
    j_j___libc_free_0(v4, 72);
  }
  return 0;
}
