// Function: sub_EFFB80
// Address: 0xeffb80
//
__int64 __fastcall sub_EFFB80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v11; // [rsp-10h] [rbp-40h]
  unsigned int v13; // [rsp+18h] [rbp-18h]

  v3 = a2;
  v4 = a3;
  v5 = 0;
  v6 = *(_DWORD *)(a1 + 1060);
  *(_DWORD *)(a1 + 1056) = 0;
  if ( !v6 )
  {
    sub_C8D5F0(a1 + 1048, (const void *)(a1 + 1064), 1u, 8u, a2, a3);
    v4 = a3;
    v3 = a2;
    v5 = 8LL * *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + v5) = 4;
  v7 = *(_QWORD *)(a1 + 1048);
  v8 = *(_QWORD *)(a1 + 1760);
  v9 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v9;
  sub_EFE900(a1 + 1576, v8, v7, v9, v3, v4, v13, 0);
  return v11;
}
