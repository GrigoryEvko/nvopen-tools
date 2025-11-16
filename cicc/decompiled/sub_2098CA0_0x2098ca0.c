// Function: sub_2098CA0
// Address: 0x2098ca0
//
void __fastcall sub_2098CA0(__int64 *a1, double a2, double a3, double a4)
{
  __int64 v4; // r12
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rsi
  int v8; // edx
  __int64 v9; // rbx
  int v10; // r13d
  __int64 v11; // [rsp+20h] [rbp-30h] BYREF
  int v12; // [rsp+28h] [rbp-28h]

  v4 = a1[69];
  if ( (*(_BYTE *)(*(_QWORD *)v4 + 808LL) & 0x10) != 0 )
  {
    v5 = *((_DWORD *)a1 + 134);
    v6 = *a1;
    v11 = 0;
    v12 = v5;
    if ( v6 )
    {
      if ( &v11 != (__int64 *)(v6 + 48) )
      {
        v7 = *(_QWORD *)(v6 + 48);
        v11 = v7;
        if ( v7 )
          sub_1623A60((__int64)&v11, v7, 2);
      }
    }
    v9 = sub_1D309E0((__int64 *)v4, 215, (__int64)&v11, 1, 0, 0, a2, a3, a4, *(_OWORD *)(v4 + 176));
    v10 = v8;
    if ( v9 )
    {
      nullsub_686();
      *(_QWORD *)(v4 + 176) = v9;
      *(_DWORD *)(v4 + 184) = v10;
      sub_1D23870();
    }
    else
    {
      *(_QWORD *)(v4 + 176) = 0;
      *(_DWORD *)(v4 + 184) = v8;
    }
    if ( v11 )
      sub_161E7C0((__int64)&v11, v11);
  }
}
