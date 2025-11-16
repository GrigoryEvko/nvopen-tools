// Function: sub_DF7C30
// Address: 0xdf7c30
//
__int64 __fastcall sub_DF7C30(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // eax
  __int64 v7; // r9
  unsigned int v8; // r13d
  unsigned __int8 *v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r9
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int8 *v16; // rdi
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // r9
  unsigned int v20; // r12d
  unsigned __int8 *v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // r9
  __int64 v24; // [rsp+8h] [rbp-38h] BYREF
  __int64 v25[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( a2 == 48 )
  {
    v18 = sub_BCB060(a4);
    v19 = *(_QWORD *)(a1 + 8);
    v20 = v18;
    v21 = *(unsigned __int8 **)(v19 + 32);
    v22 = *(_QWORD *)(v19 + 40);
    v25[0] = v18;
    return &v21[v22] == sub_DF6450(v21, (__int64)&v21[v22], v25) || v20 > (unsigned int)sub_AE43A0(v23, a3);
  }
  if ( a2 <= 0x30 )
  {
    if ( a2 == 38 )
    {
      v13 = sub_9208B0(*(_QWORD *)(a1 + 8), a3);
      v25[1] = v14;
      v25[0] = v13;
      if ( !(_BYTE)v14 )
      {
        v15 = *(_QWORD *)(a1 + 8);
        v16 = *(unsigned __int8 **)(v15 + 32);
        v17 = *(_QWORD *)(v15 + 40);
        v24 = v25[0];
        if ( &v16[v17] != sub_DF6450(v16, (__int64)&v16[v17], &v24) )
          return 0;
      }
    }
    else if ( a2 == 47 )
    {
      v6 = sub_BCB060(a3);
      v7 = *(_QWORD *)(a1 + 8);
      v8 = v6;
      v9 = *(unsigned __int8 **)(v7 + 32);
      v10 = *(_QWORD *)(v7 + 40);
      v25[0] = v6;
      if ( &v9[v10] != sub_DF6450(v9, (__int64)&v9[v10], v25) && v8 >= (unsigned int)sub_AE43A0(v11, a4) )
        return 0;
    }
    return 1;
  }
  return a2 != 49 || a3 != a4 && (*(_BYTE *)(a3 + 8) != 14 || *(_BYTE *)(a4 + 8) != 14);
}
