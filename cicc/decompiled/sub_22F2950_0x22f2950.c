// Function: sub_22F2950
// Address: 0x22f2950
//
__int64 __fastcall sub_22F2950(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r12
  _QWORD v8[2]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v9; // [rsp+10h] [rbp-90h]
  __int64 v10; // [rsp+18h] [rbp-88h]
  __int64 v11; // [rsp+20h] [rbp-80h]
  __int64 *v12; // [rsp+28h] [rbp-78h]
  __int64 v13; // [rsp+30h] [rbp-70h]
  __int64 v14; // [rsp+38h] [rbp-68h] BYREF
  __int64 v15; // [rsp+40h] [rbp-60h]
  __int64 v16; // [rsp+48h] [rbp-58h]
  __int64 v17; // [rsp+50h] [rbp-50h]
  _BYTE *v18; // [rsp+58h] [rbp-48h]
  __int64 v19; // [rsp+60h] [rbp-40h]
  _BYTE v20[56]; // [rsp+68h] [rbp-38h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_10;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v12 = &v14;
  v6 = v5 + 176;
  v18 = v20;
  v8[0] = 0;
  v8[1] = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v19 = 0;
  sub_22F1BF0(v8, a2, v5 + 176);
  sub_22EEC20(a2, v6, (__int64)v8);
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
  sub_C7D6A0(v15, 8LL * (unsigned int)v17, 8);
  if ( v12 != &v14 )
    _libc_free((unsigned __int64)v12);
  sub_C7D6A0(v9, 8LL * (unsigned int)v11, 8);
  return 0;
}
