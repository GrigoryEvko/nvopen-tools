// Function: sub_E8A5F0
// Address: 0xe8a5f0
//
_BYTE *__fastcall sub_E8A5F0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 (*v16)(void); // rdx
  char v17; // al
  _BYTE *result; // rax
  __int64 v19; // [rsp+8h] [rbp-48h] BYREF
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v21[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_E98A20(a1, a2);
  *(_QWORD *)a1 = &unk_49E2FC8;
  v9 = *a3;
  *a3 = 0;
  v21[0] = v9;
  v10 = *a5;
  *a5 = 0;
  v20 = v10;
  v11 = *a4;
  *a4 = 0;
  v19 = v11;
  v12 = sub_22077B0(376);
  v13 = v12;
  if ( v12 )
    sub_E5B9D0(v12, a2, v21, &v20, &v19);
  v14 = v19;
  *(_QWORD *)(a1 + 296) = v13;
  if ( v14 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  if ( v20 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
  if ( v21[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v21[0] + 8LL))(v21[0]);
  *(_QWORD *)(a1 + 408) = 0;
  *(_WORD *)(a1 + 304) = 1;
  *(_QWORD *)(a1 + 312) = a1 + 328;
  *(_QWORD *)(a1 + 320) = 0x200000000LL;
  v15 = *(_QWORD *)(a1 + 296);
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_DWORD *)(a1 + 432) = 0;
  v16 = *(__int64 (**)(void))(**(_QWORD **)(v15 + 8) + 16LL);
  v17 = 0;
  if ( v16 != sub_E4C920 )
    v17 = v16();
  *(_BYTE *)(a1 + 277) = v17;
  result = *(_BYTE **)(a2 + 2368);
  if ( result )
  {
    if ( (*result & 1) != 0 )
    {
      result = *(_BYTE **)(a1 + 296);
      result[33] = 1;
    }
  }
  return result;
}
