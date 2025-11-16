// Function: sub_D83CC0
// Address: 0xd83cc0
//
__int64 __fastcall sub_D83CC0(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v10; // [rsp+0h] [rbp-2E0h]
  char v11; // [rsp+1Fh] [rbp-2C1h] BYREF
  _QWORD v12[2]; // [rsp+20h] [rbp-2C0h] BYREF
  __int64 (__fastcall *v13)(_QWORD *, _QWORD *, int); // [rsp+30h] [rbp-2B0h]
  __int64 (__fastcall *v14)(__int64 *, unsigned __int64); // [rsp+38h] [rbp-2A8h]
  _QWORD v15[2]; // [rsp+40h] [rbp-2A0h] BYREF
  __int64 (__fastcall *v16)(const __m128i **, const __m128i *, int); // [rsp+50h] [rbp-290h]
  __int64 (__fastcall *v17)(__int64, unsigned __int64); // [rsp+58h] [rbp-288h]
  _BYTE v18[640]; // [rsp+60h] [rbp-280h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F87C64 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_12;
  }
  v10 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                      *(_QWORD *)(v3 + 8),
                      &unk_4F87C64)
                  + 176);
  v11 = sub_D89FA0(a2);
  v15[0] = &v11;
  v17 = sub_D75E70;
  v16 = sub_D760C0;
  v14 = sub_D75ED0;
  v15[1] = a1;
  v12[0] = a1;
  v13 = sub_D760F0;
  sub_D81040((__int64)v18, a2, (__int64)v12, v10, (__int64)v15);
  v8 = a1 + 176;
  if ( *(_BYTE *)(a1 + 760) )
  {
    *(_BYTE *)(a1 + 760) = 0;
    sub_9CD560(v8);
    v8 = a1 + 176;
  }
  sub_D77AB0(v8, (__int64)v18, v5, v6, v7, v8);
  *(_BYTE *)(a1 + 760) = 1;
  sub_9CD560((__int64)v18);
  if ( v13 )
    v13(v12, v12, 3);
  if ( v16 )
    v16((const __m128i **)v15, (const __m128i *)v15, 3);
  return 0;
}
