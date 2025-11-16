// Function: sub_20C72B0
// Address: 0x20c72b0
//
__int64 __fastcall sub_20C72B0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r9
  __int64 v9; // r14
  __int64 **v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 result; // rax
  __int64 v16; // [rsp+0h] [rbp-40h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  v8 = *(_QWORD *)(a3 + 256);
  *(_DWORD *)(a1 + 64) = 0;
  v9 = *(_QWORD *)(a3 + 248);
  *(_BYTE *)(a1 + 68) = 0;
  v10 = (__int64 **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 40) + 24LL) + 16LL * (a2 & 0x7FFFFFFF))
                   & 0xFFFFFFFFFFFFFFF8LL);
  v11 = *(_QWORD *)a4 + 24LL * *((unsigned __int16 *)*v10 + 12);
  if ( *(_DWORD *)(a4 + 8) != *(_DWORD *)v11 )
  {
    v16 = *(_QWORD *)a4 + 24LL * *((unsigned __int16 *)*v10 + 12);
    v17 = v8;
    sub_1ED7890(a4, v10);
    v11 = v16;
    v8 = v17;
  }
  v12 = *(unsigned int *)(v11 + 4);
  v13 = *(_QWORD *)(v11 + 16);
  *(_QWORD *)(a1 + 56) = v12;
  v14 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 48) = v13;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v9 + 240LL))(
         v9,
         a2,
         *(_QWORD *)(a1 + 48),
         v14,
         a1,
         v8,
         a3,
         a5) )
  {
    *(_BYTE *)(a1 + 68) = 1;
  }
  result = (unsigned int)-*(_DWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 64) = result;
  return result;
}
