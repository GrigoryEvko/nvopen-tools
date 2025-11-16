// Function: sub_3153A30
// Address: 0x3153a30
//
__int64 *__fastcall sub_3153A30(__int64 *a1, int a2, void **a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx

  v4 = sub_22077B0(0x30u);
  v5 = v4;
  if ( v4 )
  {
    *(_DWORD *)(v4 + 8) = a2;
    *(_QWORD *)v4 = &unk_49E4BC8;
    sub_CA0F50((__int64 *)(v4 + 16), a3);
  }
  *a1 = v5 | 1;
  return a1;
}
