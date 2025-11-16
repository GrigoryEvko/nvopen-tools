// Function: sub_2053B30
// Address: 0x2053b30
//
void __fastcall sub_2053B30(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v5; // r12
  char v6; // al
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rsi
  int v11; // edx
  __int64 v12; // rbx
  int v13; // r13d
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // [rsp+20h] [rbp-30h] BYREF
  int v18; // [rsp+28h] [rbp-28h]

  v5 = a1[69];
  v6 = *(_BYTE *)(*(_QWORD *)v5 + 808LL);
  if ( (v6 & 0x10) != 0 )
  {
    if ( (v6 & 0x20) != 0 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL);
      if ( !v7 || a2 != v7 - 24 )
      {
        v14 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
        v15 = v14;
        if ( !v14 )
          JUMPOUT(0x423A3A);
        if ( *(_BYTE *)(v14 - 8) == 78 )
        {
          if ( (unsigned __int8)sub_1560260((_QWORD *)(v14 + 32), -1, 29) )
            return;
          v16 = *(_QWORD *)(v15 - 48);
          if ( *(_BYTE *)(v16 + 16) )
          {
            v5 = a1[69];
          }
          else
          {
            v17 = *(_QWORD *)(v16 + 112);
            if ( (unsigned __int8)sub_1560260(&v17, -1, 29) )
              return;
            v5 = a1[69];
          }
        }
      }
    }
    v8 = *((_DWORD *)a1 + 134);
    v9 = *a1;
    v17 = 0;
    v18 = v8;
    if ( v9 )
    {
      if ( &v17 != (__int64 *)(v9 + 48) )
      {
        v10 = *(_QWORD *)(v9 + 48);
        v17 = v10;
        if ( v10 )
          sub_1623A60((__int64)&v17, v10, 2);
      }
    }
    v12 = sub_1D309E0((__int64 *)v5, 215, (__int64)&v17, 1, 0, 0, a3, a4, a5, *(_OWORD *)(v5 + 176));
    v13 = v11;
    if ( v12 )
    {
      nullsub_686();
      *(_QWORD *)(v5 + 176) = v12;
      *(_DWORD *)(v5 + 184) = v13;
      sub_1D23870();
    }
    else
    {
      *(_QWORD *)(v5 + 176) = 0;
      *(_DWORD *)(v5 + 184) = v11;
    }
    if ( v17 )
      sub_161E7C0((__int64)&v17, v17);
  }
}
