// Function: sub_26F5D00
// Address: 0x26f5d00
//
_QWORD *__fastcall sub_26F5D00(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  char v8; // r14
  __int64 i; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  char v14; // al
  _QWORD *v15; // rsi
  _QWORD *v16; // rdx
  char v17; // al
  __int64 j; // r12
  __int64 v20; // rdi
  __int64 k; // r12
  __int64 v22; // rdi
  __int64 v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  _QWORD v25[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = sub_BC0510(a4, &unk_4F82418, a3);
  v8 = *(_BYTE *)(a3 + 872);
  v23 = *(_QWORD *)(v7 + 8);
  if ( v8 )
  {
    if ( unk_4F80E08 )
    {
      sub_BA8950((__int64 *)a3);
    }
    else
    {
      for ( i = *(_QWORD *)(a3 + 32); a3 + 24 != i; i = *(_QWORD *)(v24 + 8) )
      {
        v24 = i;
        v10 = i - 56;
        if ( !i )
          v10 = 0;
        sub_B2B9A0(v10);
      }
      *(_BYTE *)(a3 + 872) = 0;
    }
  }
  v11 = sub_BC0510(a4, &unk_4F87818, a3);
  v12 = *a2;
  v13 = a2[1];
  v25[0] = v23;
  v14 = sub_26F3620(v12, v13, sub_26F14F0, v25, (__int64 *)a3, v11 + 8);
  v15 = a1 + 4;
  v16 = a1 + 10;
  if ( v14 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v15;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v16;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v15;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v16;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  v17 = *(_BYTE *)(a3 + 872);
  if ( v8 )
  {
    if ( !v17 )
    {
      for ( j = *(_QWORD *)(a3 + 32); a3 + 24 != j; j = *(_QWORD *)(j + 8) )
      {
        v20 = j - 56;
        if ( !j )
          v20 = 0;
        sub_B2B950(v20);
      }
      *(_BYTE *)(a3 + 872) = 1;
    }
  }
  else if ( v17 )
  {
    for ( k = *(_QWORD *)(a3 + 32); a3 + 24 != k; k = *(_QWORD *)(k + 8) )
    {
      v22 = k - 56;
      if ( !k )
        v22 = 0;
      sub_B2B9A0(v22);
    }
    *(_BYTE *)(a3 + 872) = 0;
  }
  return a1;
}
