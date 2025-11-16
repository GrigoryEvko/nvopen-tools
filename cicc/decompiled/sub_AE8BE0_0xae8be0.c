// Function: sub_AE8BE0
// Address: 0xae8be0
//
void __fastcall sub_AE8BE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 i; // rbx
  __int64 v13; // rax
  unsigned int v14; // eax
  _QWORD v15[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *(_BYTE *)a3 == 85 )
  {
    v13 = *(_QWORD *)(a3 - 32);
    if ( v13 )
    {
      if ( !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == *(_QWORD *)(a3 + 80) && (*(_BYTE *)(v13 + 33) & 0x20) != 0 )
      {
        v14 = *(_DWORD *)(v13 + 36);
        if ( v14 > 0x45 )
        {
          if ( v14 != 71 )
            goto LABEL_2;
          goto LABEL_16;
        }
        if ( v14 > 0x43 )
LABEL_16:
          sub_AE8A40(a1, a2, *(_QWORD *)(*(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) + 24LL));
      }
    }
  }
LABEL_2:
  v6 = *(_QWORD *)(a3 + 48);
  v15[0] = v6;
  if ( v6 )
  {
    sub_B96E90(v15, v6, 1);
    if ( v15[0] )
    {
      v7 = sub_B10CD0(v15);
      sub_AE8180(a1, a2, v7);
      if ( v15[0] )
        sub_B91220(v15);
    }
  }
  v8 = *(_QWORD *)(a3 + 64);
  if ( v8 )
  {
    v9 = sub_B14240(v8);
    v11 = v10;
    for ( i = v9; v11 != i; i = *(_QWORD *)(i + 8) )
      sub_AE8B50(a1, a2, i);
  }
}
