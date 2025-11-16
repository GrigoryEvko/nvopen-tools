// Function: sub_F460A0
// Address: 0xf460a0
//
void __fastcall sub_F460A0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v6; // zf
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 *v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 *v15; // rsi
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  _BYTE *v21; // r13
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 *v27[4]; // [rsp+0h] [rbp-20h] BYREF

  v6 = *(_BYTE *)a1 == 85;
  v27[0] = a2;
  v27[1] = a3;
  if ( v6 )
  {
    v19 = *(_QWORD *)(a1 - 32);
    if ( v19 )
    {
      if ( !*(_BYTE *)v19 )
      {
        v20 = *(_QWORD *)(a1 + 80);
        if ( *(_QWORD *)(v19 + 24) == v20 && (*(_BYTE *)(v19 + 33) & 0x20) != 0 && *(_DWORD *)(v19 + 36) == 155 )
        {
          v21 = (_BYTE *)sub_F45240(
                           v27,
                           *(__int64 **)(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + 24LL),
                           (__int64)a3,
                           v20,
                           a5,
                           a6);
          if ( v21 )
          {
            v22 = (__int64 *)sub_BD5C60(a1);
            v23 = sub_B9F6F0(v22, v21);
            v24 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
            if ( *(_QWORD *)v24 )
            {
              v25 = *(_QWORD *)(v24 + 8);
              **(_QWORD **)(v24 + 16) = v25;
              if ( v25 )
                *(_QWORD *)(v25 + 16) = *(_QWORD *)(v24 + 16);
            }
            *(_QWORD *)v24 = v23;
            if ( v23 )
            {
              v26 = *(_QWORD *)(v23 + 16);
              *(_QWORD *)(v24 + 8) = v26;
              if ( v26 )
                *(_QWORD *)(v26 + 16) = v24 + 8;
              *(_QWORD *)(v24 + 16) = v23 + 16;
              *(_QWORD *)(v23 + 16) = v24;
            }
          }
        }
      }
    }
  }
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v9 = (__int64 *)sub_B91C10(a1, 8);
    if ( v9 )
    {
      v12 = sub_F45240(v27, v9, v7, v8, v10, v11);
      if ( v12 )
        sub_B99FD0(a1, 8u, v12);
    }
    if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    {
      v15 = (__int64 *)sub_B91C10(a1, 7);
      if ( v15 )
      {
        v18 = sub_F45240(v27, v15, v13, v14, v16, v17);
        if ( v18 )
          sub_B99FD0(a1, 7u, v18);
      }
    }
  }
}
