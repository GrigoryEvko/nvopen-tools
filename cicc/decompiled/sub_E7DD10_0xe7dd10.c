// Function: sub_E7DD10
// Address: 0xe7dd10
//
__int64 __fastcall sub_E7DD10(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v9; // [rsp+8h] [rbp-28h] BYREF
  __int64 v10; // [rsp+10h] [rbp-20h] BYREF
  __int64 v11; // [rsp+18h] [rbp-18h] BYREF

  v5 = *a5;
  *a5 = 0;
  v11 = v5;
  v6 = *a4;
  *a4 = 0;
  v10 = v6;
  v7 = *a3;
  *a3 = 0;
  v9 = v7;
  sub_E8A5F0(a1, a2, &v9, &v10, &v11);
  if ( v9 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
  if ( v10 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
  if ( v11 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
  *(_BYTE *)(a1 + 6616) = 0;
  *(_QWORD *)(a1 + 3528) = a1 + 3544;
  *(_QWORD *)a1 = &unk_49E1FB0;
  *(_QWORD *)(a1 + 440) = a1 + 456;
  *(_QWORD *)(a1 + 448) = 0x4000000000LL;
  *(_QWORD *)(a1 + 3536) = 0x4000000000LL;
  return 0x4000000000LL;
}
