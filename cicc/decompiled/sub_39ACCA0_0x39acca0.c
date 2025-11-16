// Function: sub_39ACCA0
// Address: 0x39acca0
//
__int64 __fastcall sub_39ACCA0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 (*v7)(); // rdx
  __int64 *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r8
  int v12; // eax
  __int64 (__fastcall *v13)(__int64); // rax
  _BYTE v15[36]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_QWORD *)(v5 + 264);
  v7 = *(__int64 (**)())(**(_QWORD **)(v6 + 16) + 48LL);
  if ( v7 == sub_1D90020 )
  {
    v8 = 0;
  }
  else
  {
    v8 = (__int64 *)((__int64 (__fastcall *)(_QWORD))v7)(*(_QWORD *)(v6 + 16));
    v5 = *(_QWORD *)(a1 + 8);
    v6 = *(_QWORD *)(v5 + 264);
  }
  v9 = *(_QWORD *)(v5 + 240);
  v10 = *v8;
  if ( *(_DWORD *)(v9 + 348) != 4 )
    return *(_DWORD *)(a3 + 716)
         + (*(unsigned int (__fastcall **)(__int64 *, __int64, _QWORD, _BYTE *))(v10 + 176))(v8, v6, a2, v15);
  v12 = *(_DWORD *)(v9 + 352);
  if ( !v12 || v12 == 6 )
    return *(_DWORD *)(a3 + 716)
         + (*(unsigned int (__fastcall **)(__int64 *, __int64, _QWORD, _BYTE *))(v10 + 176))(v8, v6, a2, v15);
  v13 = *(__int64 (__fastcall **)(__int64))(v10 + 184);
  if ( v13 == sub_1EAD650 )
    return (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD, _BYTE *))(v10 + 176))(v8, v6, a2, v15);
  else
    return ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD, _BYTE *, __int64))v13)(v8, v6, a2, v15, 1);
}
