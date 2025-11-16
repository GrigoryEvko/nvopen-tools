// Function: sub_32590E0
// Address: 0x32590e0
//
__int64 __fastcall sub_32590E0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 (*v7)(); // rdx
  __int64 v8; // rdi
  __int64 v9; // rax
  int v11; // eax
  __int64 (__fastcall *v12)(__int64); // r9
  _DWORD v14[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_QWORD *)(v5 + 232);
  v7 = *(__int64 (**)())(**(_QWORD **)(v6 + 16) + 136LL);
  if ( v7 == sub_2DD19D0 )
  {
    v8 = 0;
  }
  else
  {
    v8 = ((__int64 (__fastcall *)(_QWORD))v7)(*(_QWORD *)(v6 + 16));
    v5 = *(_QWORD *)(a1 + 8);
    v6 = *(_QWORD *)(v5 + 232);
  }
  v9 = *(_QWORD *)(v5 + 208);
  v14[0] = 0;
  if ( *(_DWORD *)(v9 + 336) != 4 )
    return *(_DWORD *)(a3 + 748)
         + (*(unsigned int (__fastcall **)(__int64, __int64, _QWORD, _DWORD *))(*(_QWORD *)v8 + 224LL))(v8, v6, a2, v14);
  v11 = *(_DWORD *)(v9 + 344);
  if ( v11 == 6 || !v11 )
    return *(_DWORD *)(a3 + 748)
         + (*(unsigned int (__fastcall **)(__int64, __int64, _QWORD, _DWORD *))(*(_QWORD *)v8 + 224LL))(v8, v6, a2, v14);
  v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 232LL);
  if ( v12 == sub_2FDBC50 )
    return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _DWORD *))(*(_QWORD *)v8 + 224LL))(v8, v6, a2, v14);
  else
    return ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _DWORD *, __int64))v12)(v8, v6, a2, v14, 1);
}
