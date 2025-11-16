// Function: sub_278C1D0
// Address: 0x278c1d0
//
__int64 __fastcall sub_278C1D0(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v3; // r14d
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // rdi
  void *v13; // [rsp+0h] [rbp-80h] BYREF
  __int16 v14; // [rsp+20h] [rbp-60h]
  _QWORD v15[4]; // [rsp+30h] [rbp-50h] BYREF
  int v16; // [rsp+50h] [rbp-30h]
  char v17; // [rsp+54h] [rbp-2Ch]

  v1 = *(_DWORD *)(a1 + 776);
  if ( !v1 )
    return 0;
  v3 = 0;
  do
  {
    v4 = *(_QWORD *)(a1 + 768) + 16LL * v1 - 16;
    v5 = *(_QWORD *)(a1 + 24);
    v6 = *(_QWORD *)v4;
    v7 = *(_DWORD *)(v4 + 8);
    *(_DWORD *)(a1 + 776) = v1 - 1;
    v8 = *(_QWORD *)(a1 + 112);
    v14 = 257;
    v9 = *(_QWORD *)(a1 + 120);
    v15[0] = v5;
    v15[2] = v8;
    v15[1] = 0;
    v15[3] = v9;
    v16 = 0;
    v17 = 1;
    v10 = sub_F451F0(v6, v7, (__int64)v15, &v13);
    LOBYTE(v10) = v10 != 0;
    v3 |= v10;
    v1 = *(_DWORD *)(a1 + 776);
  }
  while ( v1 );
  if ( !(_BYTE)v3 )
    return 0;
  v12 = *(_QWORD *)(a1 + 16);
  if ( v12 )
    sub_102BA10(v12);
  *(_BYTE *)(a1 + 760) = 1;
  return v3;
}
