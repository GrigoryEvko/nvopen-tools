// Function: sub_ED79C0
// Address: 0xed79c0
//
__int64 *__fastcall sub_ED79C0(__int64 *a1, int a2, void *a3)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  void *v6; // [rsp+0h] [rbp-50h] BYREF
  __int16 v7; // [rsp+20h] [rbp-30h]

  v7 = 260;
  v6 = a3;
  v3 = sub_22077B0(48);
  v4 = v3;
  if ( v3 )
  {
    *(_DWORD *)(v3 + 8) = a2;
    *(_QWORD *)v3 = &unk_49E4BC8;
    sub_CA0F50((__int64 *)(v3 + 16), &v6);
  }
  *a1 = v4 | 1;
  return a1;
}
