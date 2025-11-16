// Function: sub_29E0FA0
// Address: 0x29e0fa0
//
void __fastcall sub_29E0FA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 j; // rbx
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 i; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return;
  v3 = sub_B91C10(a1, 10);
  v4 = v3;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
  {
    v19 = 0;
LABEL_34:
    v6 = 0;
    v7 = 0;
    goto LABEL_6;
  }
  v19 = sub_B91C10(a1, 25);
  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
  {
    v3 = v4 | v19;
    goto LABEL_34;
  }
  v5 = sub_B91C10(a1, 7);
  v6 = v5;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v7 = sub_B91C10(a1, 8);
    v3 = v7 | v6 | v4 | v19;
  }
  else
  {
    v7 = 0;
    v3 = v5 | v4 | v19;
  }
LABEL_6:
  if ( v3 )
  {
    for ( i = a2; a3 != i; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
        BUG();
      for ( j = *(_QWORD *)(i + 32); i + 24 != j; j = *(_QWORD *)(j + 8) )
      {
        v9 = j - 24;
        if ( !j )
          v9 = 0;
        if ( (unsigned __int8)sub_B46420(v9) || (unsigned __int8)sub_B46490(v9) )
        {
          if ( v4 )
          {
            v10 = 0;
            if ( (*(_BYTE *)(v9 + 7) & 0x20) != 0 )
              v10 = sub_B91C10(v9, 10);
            v4 = sub_BA72D0(v10, v4);
            sub_B99FD0(v9, 0xAu, v4);
          }
          if ( v19 )
          {
            v11 = 0;
            if ( (*(_BYTE *)(v9 + 7) & 0x20) != 0 )
              v11 = sub_B91C10(v9, 25);
            v12 = sub_9C14F0(v11, v19);
            sub_B99FD0(v9, 0x19u, v12);
          }
          if ( v6 )
          {
            v13 = 0;
            if ( (*(_BYTE *)(v9 + 7) & 0x20) != 0 )
              v13 = sub_B91C10(v9, 7);
            v14 = sub_BA72D0(v13, v6);
            sub_B99FD0(v9, 7u, v14);
          }
          if ( v7 )
          {
            v15 = 0;
            if ( (*(_BYTE *)(v9 + 7) & 0x20) != 0 )
              v15 = sub_B91C10(v9, 8);
            v16 = sub_BA72D0(v15, v7);
            sub_B99FD0(v9, 8u, v16);
          }
        }
      }
    }
  }
}
