// Function: sub_F02830
// Address: 0xf02830
//
__int64 __fastcall sub_F02830(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // rdx
  bool v8; // zf
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v15[2]; // [rsp+0h] [rbp-790h] BYREF
  _BYTE v16[1808]; // [rsp+10h] [rbp-780h] BYREF
  char v17; // [rsp+720h] [rbp-70h]
  __int64 v18; // [rsp+728h] [rbp-68h]
  __int64 v19; // [rsp+730h] [rbp-60h]
  bool v20; // [rsp+738h] [rbp-58h]
  char v21; // [rsp+750h] [rbp-40h]

  v6 = a1 + 176;
  if ( !*(_BYTE *)(a1 + 168) )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = *(_DWORD *)(a1 + 1904) == 2;
    v9 = 0;
    v17 = 0;
    if ( v8 )
      v9 = a1 + 32;
    v20 = v8;
    v15[1] = v7;
    v18 = a1 + 176;
    v19 = v9;
    v15[0] = (__int64)&unk_49E4EE8;
    v21 = 0;
    sub_F027B0(v15, (__int64)a2, (__int64)&unk_49E4ED8, a4, a5, a6);
    v8 = v17 == 0;
    *(_BYTE *)(a1 + 168) = 1;
    v15[0] = (__int64)&unk_49E4EE8;
    if ( !v8 )
    {
      v17 = 0;
      sub_EFDF60((__int64)v16, a2, v10, v11, v12, v13);
    }
  }
  sub_EFFD30(v6, (int *)a2, a1 + 32);
  return sub_EFDF30(v6, *(_QWORD *)(a1 + 16));
}
