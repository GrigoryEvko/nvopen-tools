// Function: sub_2466140
// Address: 0x2466140
//
__int64 __fastcall sub_2466140(__int64 *a1, __int64 a2, _BYTE *a3, _DWORD *a4, __int64 a5, __int64 a6)
{
  _DWORD *v6; // r10
  __int64 (*v9)(void); // rax
  __int64 v10; // rax
  __int64 v11; // r12
  _QWORD *v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // rax
  const void *v20; // [rsp+0h] [rbp-70h]
  _BYTE v23[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v24; // [rsp+30h] [rbp-40h]

  v6 = a4;
  v9 = *(__int64 (**)(void))(*(_QWORD *)a1[10] + 88LL);
  if ( (char *)v9 != (char *)sub_9482E0 )
  {
    v18 = v9();
    v6 = a4;
    v11 = v18;
LABEL_5:
    if ( v11 )
      return v11;
    goto LABEL_7;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *a3 <= 0x15u )
  {
    v10 = sub_AAAE30(a2, (__int64)a3, a4, a5);
    v6 = a4;
    v11 = v10;
    goto LABEL_5;
  }
LABEL_7:
  v20 = v6;
  v24 = 257;
  v13 = sub_BD2C40(104, unk_3F148BC);
  v11 = (__int64)v13;
  if ( v13 )
  {
    sub_B44260((__int64)v13, *(_QWORD *)(a2 + 8), 65, 2u, 0, 0);
    *(_QWORD *)(v11 + 72) = v11 + 88;
    *(_QWORD *)(v11 + 80) = 0x400000000LL;
    sub_B4FD20(v11, a2, (__int64)a3, v20, a5, (__int64)v23);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v11,
    a6,
    a1[7],
    a1[8]);
  v14 = *a1;
  v15 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v15 )
  {
    do
    {
      v16 = *(_QWORD *)(v14 + 8);
      v17 = *(_DWORD *)v14;
      v14 += 16;
      sub_B99FD0(v11, v17, v16);
    }
    while ( v15 != v14 );
  }
  return v11;
}
