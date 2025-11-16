// Function: sub_3111350
// Address: 0x3111350
//
__int64 *__fastcall sub_3111350(__int64 *a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  void *v5; // [rsp+0h] [rbp-50h] BYREF
  __int16 v6; // [rsp+20h] [rbp-30h]

  v6 = 257;
  v2 = sub_22077B0(0x30u);
  v3 = v2;
  if ( v2 )
  {
    *(_DWORD *)(v2 + 8) = a2;
    *(_QWORD *)v2 = &unk_4A32A78;
    sub_CA0F50((__int64 *)(v2 + 16), &v5);
  }
  *a1 = v3 | 1;
  return a1;
}
