// Function: sub_39158C0
// Address: 0x39158c0
//
__int64 __fastcall sub_39158C0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  char v6; // di
  unsigned int v7; // edx
  __int64 v8; // rcx
  __int64 v9; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rbx
  __int64 v13; // rdi
  unsigned __int32 v14; // r15d
  unsigned __int32 v15; // ebx
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rax
  unsigned __int32 v19; // ecx
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rsi
  char v29[52]; // [rsp+Ch] [rbp-34h] BYREF

  v4 = *a2;
  v5 = a2[1];
  v6 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) & 1;
  if ( v4 == v5 )
  {
    v9 = 11;
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
    v9 = v7 - 1LL;
  }
  v10 = (-(__int64)(v6 == 0) & 0xFFFFFFFFFFFFFFFCLL) + 8;
  v11 = (v10 + v9) % v10;
  v12 = (v10 + v9) / v10 * v10;
  (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64))(**(_QWORD **)(a1 + 240) + 64LL))(
    *(_QWORD *)(a1 + 240),
    v5,
    v11);
  v13 = *(_QWORD *)(a1 + 240);
  v14 = v12;
  v15 = _byteswap_ulong(v12);
  *(_DWORD *)v29 = (unsigned int)(*(_DWORD *)(a1 + 248) - 1) < 2 ? 45 : 754974720;
  sub_16E7EE0(v13, v29, 4u);
  v16 = *(_QWORD *)(a1 + 240);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v14 = v15;
  *(_DWORD *)v29 = v14;
  sub_16E7EE0(v16, v29, 4u);
  v17 = *(_QWORD *)(a1 + 240);
  v18 = (a2[1] - *a2) >> 5;
  v19 = _byteswap_ulong(v18);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    LODWORD(v18) = v19;
  *(_DWORD *)v29 = v18;
  sub_16E7EE0(v17, v29, 4u);
  v20 = *a2;
  v21 = a2[1];
  if ( *a2 == v21 )
  {
    v26 = 11;
    LODWORD(v22) = 12;
  }
  else
  {
    v22 = 12;
    do
    {
      v24 = sub_16E7EE0(*(_QWORD *)(a1 + 240), *(char **)v20, *(_QWORD *)(v20 + 8));
      v25 = *(_BYTE **)(v24 + 24);
      if ( (unsigned __int64)v25 < *(_QWORD *)(v24 + 16) )
      {
        *(_QWORD *)(v24 + 24) = v25 + 1;
        *v25 = 0;
      }
      else
      {
        sub_16E7DE0(v24, 0);
      }
      v23 = *(_QWORD *)(v20 + 8);
      v20 += 32;
      v22 += v23 + 1;
    }
    while ( v21 != v20 );
    v26 = v22 - 1;
  }
  v27 = (-(__int64)((*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) & 1) == 0) & 0xFFFFFFFFFFFFFFFCLL) + 8;
  return sub_16E8900(*(_QWORD *)(a1 + 240), v27 * ((v27 + v26) / v27) - v22);
}
