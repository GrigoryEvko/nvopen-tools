// Function: sub_2F3F000
// Address: 0x2f3f000
//
__int64 __fastcall sub_2F3F000(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v8; // [rsp+0h] [rbp-50h] BYREF
  __int64 v9; // [rsp+8h] [rbp-48h] BYREF
  __int64 v10[5]; // [rsp+10h] [rbp-40h] BYREF
  char v11; // [rsp+38h] [rbp-18h]

  v2 = *(__int64 **)(a1 + 8);
  v8 = a1;
  v9 = a1;
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5027190 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_6;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_5027190);
  v11 = 1;
  v10[0] = *(_QWORD *)(v5 + 256);
  v10[1] = (__int64)sub_2F3C380;
  v10[2] = (__int64)&v8;
  v10[3] = (__int64)sub_2F3C110;
  v10[4] = (__int64)&v9;
  return sub_2F3E6C0(v10, a2, v6);
}
