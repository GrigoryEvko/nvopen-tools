// Function: sub_2053930
// Address: 0x2053930
//
void __fastcall sub_2053930(__int64 *a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rsi
  __int64 v5; // r14
  int v6; // r13d
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // [rsp+20h] [rbp-60h] BYREF
  int v10; // [rsp+28h] [rbp-58h]
  char v11; // [rsp+30h] [rbp-50h] BYREF
  __int64 v12; // [rsp+40h] [rbp-40h]
  __int64 v13; // [rsp+48h] [rbp-38h]

  v1 = a1[69];
  v2 = *a1;
  v9 = 0;
  v3 = *(_QWORD *)(v1 + 16);
  v10 = *((_DWORD *)a1 + 134);
  if ( v2 )
  {
    if ( &v9 != (__int64 *)(v2 + 48) )
    {
      v4 = *(_QWORD *)(v2 + 48);
      v9 = v4;
      if ( v4 )
      {
        sub_1623A60((__int64)&v9, v4, 2);
        v1 = a1[69];
      }
    }
  }
  sub_20BE530((unsigned int)&v11, v3, v1, 460, 112, 0, 0, 0, 0, (__int64)&v9, 0, 0);
  v5 = v12;
  v6 = v13;
  v7 = v12;
  if ( v9 )
    sub_161E7C0((__int64)&v9, v9);
  v8 = a1[69];
  if ( v7 )
  {
    nullsub_686();
    *(_QWORD *)(v8 + 176) = v5;
    *(_DWORD *)(v8 + 184) = v6;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v8 + 176) = 0;
    *(_DWORD *)(v8 + 184) = v6;
  }
}
