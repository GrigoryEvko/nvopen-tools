// Function: sub_278C0E0
// Address: 0x278c0e0
//
__int64 __fastcall sub_278C0E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rsi
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // r13
  int v11; // eax
  __int64 v12; // r13
  unsigned int i; // r12d
  __int64 v14; // r12
  __int64 v15; // rdi
  _QWORD v17[4]; // [rsp+0h] [rbp-80h] BYREF
  int v18; // [rsp+20h] [rbp-60h]
  char v19; // [rsp+24h] [rbp-5Ch]
  void *v20; // [rsp+30h] [rbp-50h] BYREF
  __int16 v21; // [rsp+50h] [rbp-30h]

  v3 = (__int64 *)(a2 + 48);
  v6 = *v3;
  v7 = *(_QWORD *)(a1 + 120);
  v19 = 0;
  v8 = *(_QWORD *)(a1 + 112);
  v9 = *(_QWORD *)(a1 + 24);
  v17[1] = 0;
  v10 = v6 & 0xFFFFFFFFFFFFFFF8LL;
  v17[3] = v7;
  v17[0] = v9;
  v17[2] = v8;
  v18 = 0;
  if ( (__int64 *)v10 == v3 )
  {
    v12 = 0;
  }
  else
  {
    if ( !v10 )
      BUG();
    v11 = *(unsigned __int8 *)(v10 - 24);
    v12 = v10 - 24;
    if ( (unsigned int)(v11 - 30) >= 0xB )
      v12 = 0;
  }
  for ( i = 0; a3 != sub_B46EC0(v12, i); ++i )
    ;
  v21 = 257;
  v14 = sub_F451F0(v12, i, (__int64)v17, &v20);
  if ( v14 )
  {
    v15 = *(_QWORD *)(a1 + 16);
    if ( v15 )
      sub_102BA10(v15);
    *(_BYTE *)(a1 + 760) = 1;
  }
  sub_278C0C0(a1);
  return v14;
}
