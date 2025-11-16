// Function: sub_AEC060
// Address: 0xaec060
//
void __fastcall sub_AEC060(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v14[0] = a1;
  v3 = *(_QWORD *)(a2 + 64);
  if ( v3 )
  {
    v4 = sub_B14240(v3);
    v6 = v5;
    v7 = v4;
    if ( v4 != v5 )
    {
      while ( *(_BYTE *)(v7 + 32) )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( v7 == v5 )
          goto LABEL_11;
      }
      if ( v7 != v5 )
      {
        if ( *(_BYTE *)(v7 + 64) != 2 )
          goto LABEL_10;
LABEL_17:
        v10 = sub_B13870(v7);
        v11 = sub_AEBD40(v14, v10);
        sub_B13D10(v7, v11);
LABEL_10:
        while ( 1 )
        {
          v7 = *(_QWORD *)(v7 + 8);
          if ( v7 == v6 )
            break;
          if ( !*(_BYTE *)(v7 + 32) )
          {
            if ( v6 == v7 )
              break;
            if ( *(_BYTE *)(v7 + 64) == 2 )
              goto LABEL_17;
          }
        }
      }
    }
  }
LABEL_11:
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && (v8 = sub_B91C10(a2, 38)) != 0 )
  {
    v9 = sub_AEBD40(v14, v8);
    sub_B99FD0(a2, 38, v9);
  }
  else if ( *(_BYTE *)a2 == 85 )
  {
    v12 = *(_QWORD *)(a2 - 32);
    if ( v12 )
    {
      if ( !*(_BYTE *)v12
        && *(_QWORD *)(v12 + 24) == *(_QWORD *)(a2 + 80)
        && (*(_BYTE *)(v12 + 33) & 0x20) != 0
        && *(_DWORD *)(v12 + 36) == 68 )
      {
        v13 = sub_AEBD40(v14, *(_QWORD *)(*(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL));
        sub_B59600(a2, v13);
      }
    }
  }
}
