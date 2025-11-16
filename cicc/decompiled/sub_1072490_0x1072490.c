// Function: sub_1072490
// Address: 0x1072490
//
__int64 __fastcall sub_1072490(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  __int64 v5; // rsi
  char v6; // di
  int v7; // eax
  __int64 v8; // rcx
  _BOOL4 v9; // ebx
  unsigned int v10; // eax
  unsigned int v11; // ecx
  unsigned int v12; // ebx
  int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  unsigned int v17; // edx
  __int64 v18; // rbx
  __int64 v19; // r13
  int v20; // r12d
  __int64 v21; // rax
  __int64 v22; // rdi
  _BYTE *v23; // rax
  int v24; // ecx
  unsigned __int8 v26[36]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *a2;
  v5 = a2[1];
  v6 = *(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1;
  if ( v4 == v5 )
  {
    v10 = 11;
    v9 = 1;
  }
  else
  {
    v7 = 12;
    do
    {
      v8 = *(_QWORD *)(v4 + 8);
      v4 += 32;
      v7 += v8 + 1;
    }
    while ( v5 != v4 );
    v9 = v7 != 0;
    v10 = v7 - v9;
  }
  v11 = v6 == 0 ? 4 : 8;
  v12 = v11 * (v10 / v11 + v9);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 2048) + 80LL))(
    *(_QWORD *)(a1 + 2048),
    v5,
    v10 % v11);
  v13 = 45;
  v14 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v13 = 754974720;
  *(_DWORD *)v26 = v13;
  sub_CB6200(v14, v26, 4u);
  v15 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v12 = _byteswap_ulong(v12);
  *(_DWORD *)v26 = v12;
  sub_CB6200(v15, v26, 4u);
  v16 = *(_QWORD *)(a1 + 2048);
  v17 = (a2[1] - *a2) >> 5;
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v17 = _byteswap_ulong(v17);
  *(_DWORD *)v26 = v17;
  sub_CB6200(v16, v26, 4u);
  v18 = *a2;
  v19 = a2[1];
  if ( *a2 == v19 )
  {
    v24 = 11;
    v20 = 12;
  }
  else
  {
    v20 = 12;
    do
    {
      v22 = sub_CB6200(*(_QWORD *)(a1 + 2048), *(unsigned __int8 **)v18, *(_QWORD *)(v18 + 8));
      v23 = *(_BYTE **)(v22 + 32);
      if ( (unsigned __int64)v23 < *(_QWORD *)(v22 + 24) )
      {
        *(_QWORD *)(v22 + 32) = v23 + 1;
        *v23 = 0;
      }
      else
      {
        sub_CB5D20(v22, 0);
      }
      v21 = *(_QWORD *)(v18 + 8);
      v18 += 32;
      v20 += v21 + 1;
    }
    while ( v19 != v18 );
    v24 = v20 - 1;
  }
  return sub_CB6C70(
           *(_QWORD *)(a1 + 2048),
           (((*(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1) == 0 ? -4 : -8)
          & (v24 + ((*(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1) == 0 ? 0xFFFFFFFC : 0) + 8))
         - v20);
}
