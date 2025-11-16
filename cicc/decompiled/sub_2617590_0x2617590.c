// Function: sub_2617590
// Address: 0x2617590
//
__int64 __fastcall sub_2617590(__int64 *a1, unsigned __int64 a2)
{
  _BYTE *v2; // rbx
  __int64 v3; // rdi
  _BYTE v5[8]; // [rsp+0h] [rbp-20h] BYREF
  __int64 v6; // [rsp+8h] [rbp-18h]

  v2 = (_BYTE *)a1[1];
  sub_B8BA60((__int64)v5, *(_QWORD *)(*a1 + 8), *a1, (__int64)&unk_4F875EC, a2);
  v3 = v6;
  if ( v2 )
    *v2 |= v5[0];
  return (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_4F875EC) + 176;
}
