// Function: sub_1353760
// Address: 0x1353760
//
__int64 __fastcall sub_1353760(__int64 a1, const char *a2, __int64 a3)
{
  _QWORD *v3; // r12

  v3 = *(_QWORD **)(a1 + 160);
  *(_QWORD *)(a1 + 160) = 0;
  if ( v3 )
  {
    sub_13525A0(v3, a2, a3);
    j_j___libc_free_0(v3, 104);
  }
  return 0;
}
