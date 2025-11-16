// Function: sub_2729D90
// Address: 0x2729d90
//
__int64 __fastcall sub_2729D90(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  const char *v7; // rax
  const char *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  const char *v13[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v14; // [rsp+20h] [rbp-40h]

  v6 = sub_B47F80(a1);
  v7 = sub_BD5D20((__int64)a1);
  v13[1] = v8;
  v14 = 261;
  v13[0] = v7;
  sub_BD6B50((unsigned __int8 *)v6, v13);
  sub_B44220((_QWORD *)v6, a2, a3);
  if ( a4 )
  {
    if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
      v9 = *(_QWORD *)(v6 - 8);
    else
      v9 = v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
    if ( *(_QWORD *)v9 )
    {
      v10 = *(_QWORD *)(v9 + 8);
      **(_QWORD **)(v9 + 16) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v9 + 16);
    }
    *(_QWORD *)v9 = a4;
    v11 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(v9 + 8) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = v9 + 8;
    *(_QWORD *)(v9 + 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = v9;
  }
  return v6;
}
