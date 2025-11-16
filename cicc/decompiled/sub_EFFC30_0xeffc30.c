// Function: sub_EFFC30
// Address: 0xeffc30
//
__int64 __fastcall sub_EFFC30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // ecx
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v14; // [rsp-10h] [rbp-40h]
  unsigned int v15; // [rsp+8h] [rbp-28h]

  v6 = 0;
  v7 = *(_DWORD *)(a1 + 1060);
  *(_DWORD *)(a1 + 1056) = 0;
  if ( !v7 )
  {
    sub_C8D5F0(a1 + 1048, (const void *)(a1 + 1064), 1u, 8u, a5, a6);
    v6 = 8LL * *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + v6) = 2;
  v8 = *(unsigned int *)(a1 + 1060);
  v9 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v9;
  if ( v9 + 1 > v8 )
  {
    sub_C8D5F0(a1 + 1048, (const void *)(a1 + 1064), v9 + 1, 8u, a5, a6);
    v9 = *(unsigned int *)(a1 + 1056);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1048) + 8 * v9) = a2;
  v10 = *(_QWORD *)(a1 + 1048);
  v11 = *(_QWORD *)(a1 + 1744);
  v12 = (unsigned int)(*(_DWORD *)(a1 + 1056) + 1);
  *(_DWORD *)(a1 + 1056) = v12;
  sub_EFE900(a1 + 1576, v11, v10, v12, 0, 0, v15, 0);
  return v14;
}
