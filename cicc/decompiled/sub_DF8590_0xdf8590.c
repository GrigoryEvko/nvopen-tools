// Function: sub_DF8590
// Address: 0xdf8590
//
__int64 __fastcall sub_DF8590(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  unsigned int v7; // r12d
  unsigned __int8 *v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // r9
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int8 *v14; // rdi
  __int64 v15; // rsi
  unsigned int v16; // r13d
  unsigned __int8 *v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // r9
  __int64 v20; // [rsp+8h] [rbp-38h] BYREF
  __int64 v21[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( a2 == 48 )
  {
    v16 = sub_BCB060(a4);
    v17 = *(unsigned __int8 **)(*a1 + 32);
    v18 = *(_QWORD *)(*a1 + 40);
    v21[0] = v16;
    if ( &v17[v18] == sub_DF6450(v17, (__int64)&v17[v18], v21) || (unsigned int)sub_AE43A0(v19, a3) < v16 )
      return 1;
  }
  else if ( a2 > 0x30 )
  {
    if ( a2 != 49 || a3 != a4 && (*(_BYTE *)(a3 + 8) != 14 || *(_BYTE *)(a4 + 8) != 14) )
      return 1;
  }
  else if ( a2 == 38 )
  {
    v12 = sub_9208B0(*a1, a3);
    v21[1] = v13;
    v21[0] = v12;
    if ( (_BYTE)v13 )
      return 1;
    v14 = *(unsigned __int8 **)(*a1 + 32);
    v15 = *(_QWORD *)(*a1 + 40);
    v20 = v21[0];
    if ( &v14[v15] == sub_DF6450(v14, (__int64)&v14[v15], &v20) )
      return 1;
  }
  else
  {
    if ( a2 != 47 )
      return 1;
    v7 = sub_BCB060(a3);
    v8 = *(unsigned __int8 **)(*a1 + 32);
    v9 = *(_QWORD *)(*a1 + 40);
    v21[0] = v7;
    if ( &v8[v9] == sub_DF6450(v8, (__int64)&v8[v9], v21) || (unsigned int)sub_AE43A0(v10, a4) > v7 )
      return 1;
  }
  return 0;
}
