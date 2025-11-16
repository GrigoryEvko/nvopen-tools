// Function: sub_38D3FD0
// Address: 0x38d3fd0
//
__int64 __fastcall sub_38D3FD0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rdi
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  __int64 v17; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v18[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_38DCAE0();
  *(_QWORD *)a1 = &unk_4A3E0F8;
  v9 = *a3;
  *a3 = 0;
  v18[0] = v9;
  v10 = *a5;
  *a5 = 0;
  v17 = v10;
  v11 = *a4;
  *a4 = 0;
  v16 = v11;
  v12 = sub_22077B0(0x850u);
  v13 = v12;
  if ( v12 )
    sub_390A820(v12, a2, v18, &v17, &v16);
  v14 = v16;
  *(_QWORD *)(a1 + 264) = v13;
  if ( v14 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  if ( v17 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
  if ( v18[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v18[0] + 8LL))(v18[0]);
  *(_QWORD *)(a1 + 272) = 0;
  *(_WORD *)(a1 + 280) = 1;
  *(_QWORD *)(a1 + 288) = a1 + 304;
  *(_QWORD *)(a1 + 296) = 0x200000000LL;
  return 0x200000000LL;
}
