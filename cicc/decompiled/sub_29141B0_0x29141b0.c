// Function: sub_29141B0
// Address: 0x29141b0
//
__int64 __fastcall sub_29141B0(_QWORD **a1, _QWORD **a2, __int64 a3)
{
  _QWORD **v3; // rbx
  __int64 v4; // r14
  _QWORD *v5; // r13
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int8 v12; // dl
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  __int64 v21[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( a1 != a2 )
  {
    v3 = a1;
    v4 = a3 + 72;
    do
    {
      while ( 1 )
      {
        v5 = *v3;
        v6 = sub_B12000((__int64)(*v3 + 9));
        if ( v6 == sub_B12000(v4) )
          break;
LABEL_3:
        if ( a2 == ++v3 )
          return a3;
      }
      v7 = v5[3];
      v20 = v7;
      if ( v7 )
        sub_B96E90((__int64)&v20, v7, 1);
      v8 = sub_B10CD0((__int64)&v20);
      v9 = *(_BYTE *)(v8 - 16);
      if ( (v9 & 2) != 0 )
      {
        if ( *(_DWORD *)(v8 - 24) != 2 )
          goto LABEL_9;
        v14 = *(_QWORD *)(v8 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v8 - 16) >> 6) & 0xF) != 2 )
        {
LABEL_9:
          v17 = 0;
          goto LABEL_10;
        }
        v14 = v8 - 16 - 8LL * ((v9 >> 2) & 0xF);
      }
      v17 = *(_QWORD *)(v14 + 8);
LABEL_10:
      v10 = *(_QWORD *)(a3 + 24);
      v21[0] = v10;
      if ( v10 )
        sub_B96E90((__int64)v21, v10, 1);
      v11 = sub_B10CD0((__int64)v21);
      v12 = *(_BYTE *)(v11 - 16);
      if ( (v12 & 2) != 0 )
      {
        if ( *(_DWORD *)(v11 - 24) != 2 )
          goto LABEL_14;
        v15 = *(_QWORD *)(v11 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v11 - 16) >> 6) & 0xF) != 2 )
        {
LABEL_14:
          v18 = 0;
          goto LABEL_15;
        }
        v15 = v11 - 16 - 8LL * ((v12 >> 2) & 0xF);
      }
      v18 = *(_QWORD *)(v15 + 8);
LABEL_15:
      if ( v21[0] )
        sub_B91220((__int64)v21, v21[0]);
      if ( v20 )
        sub_B91220((__int64)&v20, v20);
      if ( v18 != v17 )
        goto LABEL_3;
      ++v3;
      sub_B14290(v5);
    }
    while ( a2 != v3 );
  }
  return a3;
}
