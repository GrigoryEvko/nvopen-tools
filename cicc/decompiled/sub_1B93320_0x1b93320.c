// Function: sub_1B93320
// Address: 0x1b93320
//
__int64 __fastcall sub_1B93320(__int64 a1, __int64 a2, int *a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  char v7; // bl
  __int64 result; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v10)(const __m128i **, const __m128i *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v11)(); // [rsp+18h] [rbp-28h]

  v4 = sub_1B8F3E0(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 384LL), a2);
  if ( !v4 )
    return 0;
  v9[0] = a1;
  v6 = v4;
  v9[1] = a2;
  v11 = sub_1B99770;
  v10 = sub_1B8E1C0;
  v7 = sub_1B932A0((__int64)v9, a3, v5);
  if ( v10 )
    v10((const __m128i **)v9, (const __m128i *)v9, 3);
  if ( !v7 )
    return 0;
  result = sub_22077B0(48);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = 0;
    *(_BYTE *)(result + 24) = 3;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)result = &unk_49F6F78;
    *(_QWORD *)(result + 40) = v6;
  }
  return result;
}
