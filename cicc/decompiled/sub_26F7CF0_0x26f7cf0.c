// Function: sub_26F7CF0
// Address: 0x26f7cf0
//
__int64 __fastcall sub_26F7CF0(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 i; // r12
  unsigned __int8 v14; // [rsp-4Dh] [rbp-4Dh] BYREF
  int v15; // [rsp-4Ch] [rbp-4Ch] BYREF
  __int64 *v16[9]; // [rsp-48h] [rbp-48h] BYREF

  v6 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = *(__int64 **)(v6 + 24);
  if ( v7 == *(__int64 **)(v6 + 32) )
    return 0;
  v16[0] = a2;
  v16[1] = (__int64 *)&v15;
  v14 = 0;
  v15 = 3;
  v16[2] = (__int64 *)&v14;
  v16[3] = v7;
  sub_26F7B50(v16, a1, (__int64)v7, a4, a5, a6);
  for ( i = a1[13]; a1 + 11 != (_QWORD *)i; i = sub_220EEE0(i) )
    sub_26F7B50(v16, (_QWORD *)(i + 56), v9, v10, v11, v12);
  return v14;
}
