// Function: sub_3906860
// Address: 0x3906860
//
__int64 __fastcall sub_3906860(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r13d
  char v5; // cl
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdi
  const char *v13; // rax
  __int64 v14; // rdi
  _QWORD v16[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v17[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v18; // [rsp+20h] [rbp-40h]

  if ( a3 != 5 )
  {
    switch ( a3 )
    {
      case 6LL:
        if ( *(_DWORD *)a2 != 1668246574 )
          goto LABEL_13;
        v4 = 13;
        if ( *(_WORD *)(a2 + 4) != 27745 )
          goto LABEL_13;
        goto LABEL_14;
      case 7LL:
        if ( *(_DWORD *)a2 != 1684629550 )
          goto LABEL_13;
        if ( *(_WORD *)(a2 + 4) != 25956 )
          goto LABEL_13;
        v4 = 9;
        if ( *(_BYTE *)(a2 + 6) != 110 )
          goto LABEL_13;
        goto LABEL_14;
      case 9LL:
        if ( *(_QWORD *)a2 == 0x616E7265746E692ELL )
        {
          v4 = 11;
          if ( *(_BYTE *)(a2 + 8) == 108 )
            goto LABEL_14;
        }
        goto LABEL_13;
    }
    goto LABEL_9;
  }
  if ( *(_DWORD *)a2 != 1634039598 || (v5 = 0, v6 = 1, *(_BYTE *)(a2 + 4) != 107) )
  {
LABEL_9:
    v5 = 1;
    v6 = 0;
  }
  if ( a3 != 10 || !v5 )
  {
    v4 = 20;
    if ( v6 )
      goto LABEL_14;
    goto LABEL_13;
  }
  if ( *(_QWORD *)a2 != 0x746365746F72702ELL || (v4 = 18, *(_WORD *)(a2 + 8) != 25701) )
LABEL_13:
    v4 = 0;
LABEL_14:
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
LABEL_22:
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    return 0;
  }
  else
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(a1 + 8);
      v16[0] = 0;
      v16[1] = 0;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v12 + 144LL))(v12, v16) )
      {
        HIBYTE(v18) = 1;
        v13 = "expected identifier in directive";
        goto LABEL_21;
      }
      v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v17[0] = v16;
      v18 = 261;
      v8 = sub_38BF510(v7, (__int64)v17);
      v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v9 + 256LL))(v9, v8, v4);
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
        goto LABEL_22;
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
        break;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    }
    HIBYTE(v18) = 1;
    v13 = "unexpected token in directive";
LABEL_21:
    v14 = *(_QWORD *)(a1 + 8);
    v17[0] = v13;
    LOBYTE(v18) = 3;
    return sub_3909CF0(v14, v17, 0, 0, v10, v11);
  }
}
