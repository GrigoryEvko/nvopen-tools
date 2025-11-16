// Function: sub_D89AB0
// Address: 0xd89ab0
//
__int64 __fastcall sub_D89AB0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  _QWORD v6[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 (__fastcall *v7)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-60h]
  __int64 (__fastcall *v8)(__int64); // [rsp+18h] [rbp-58h]
  _QWORD v9[10]; // [rsp+20h] [rbp-50h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F881C8 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_8;
  }
  v6[0] = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                        *(_QWORD *)(v3 + 8),
                        &unk_4F881C8)
                    + 176);
  v8 = sub_D853D0;
  v7 = sub_D857F0;
  sub_D898E0(v9, a2, (__int64)v6);
  sub_D89990(a1 + 176, (__int64)v9);
  sub_D89A50((__int64)v9);
  if ( v7 )
    v7(v6, v6, 3);
  return 0;
}
