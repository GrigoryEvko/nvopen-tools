// Function: sub_ECB010
// Address: 0xecb010
//
__int64 __fastcall sub_ECB010(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  __int64 v5; // rdi
  __int64 *v6; // rdi
  __int64 v7; // rax
  __int64 (*v8)(); // rcx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  bool v12; // zf
  __int64 v14; // rdi
  __int64 v15; // rdi
  const char *v16; // [rsp+0h] [rbp-70h] BYREF
  const char *v17; // [rsp+8h] [rbp-68h]
  const char *v18[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v19; // [rsp+30h] [rbp-40h]

  switch ( a3 )
  {
    case 5LL:
      if ( *(_DWORD *)a2 == 1634039598 )
      {
        v4 = 24;
        if ( *(_BYTE *)(a2 + 4) == 107 )
          break;
      }
LABEL_3:
      v4 = 0;
      break;
    case 6LL:
      if ( *(_DWORD *)a2 != 1668246574 )
        goto LABEL_3;
      v4 = 17;
      if ( *(_WORD *)(a2 + 4) != 27745 )
        goto LABEL_3;
      break;
    case 7LL:
      if ( *(_DWORD *)a2 != 1684629550 )
        goto LABEL_3;
      if ( *(_WORD *)(a2 + 4) != 25956 )
        goto LABEL_3;
      v4 = 12;
      if ( *(_BYTE *)(a2 + 6) != 110 )
        goto LABEL_3;
      break;
    case 9LL:
      if ( *(_QWORD *)a2 != 0x616E7265746E692ELL )
        goto LABEL_3;
      v4 = 15;
      if ( *(_BYTE *)(a2 + 8) != 108 )
        goto LABEL_3;
      break;
    default:
      if ( a3 == 10 && *(_QWORD *)a2 == 0x746365746F72702ELL && *(_WORD *)(a2 + 8) == 25701 )
      {
        v4 = 22;
        break;
      }
      goto LABEL_3;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
LABEL_21:
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    return 0;
  }
  else
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)(a1 + 8);
        v16 = 0;
        v17 = 0;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v5 + 192LL))(v5, &v16) )
        {
          v14 = *(_QWORD *)(a1 + 8);
          v18[0] = "expected identifier";
          v19 = 259;
          return sub_ECE0E0(v14, v18, 0, 0);
        }
        v6 = *(__int64 **)(a1 + 8);
        v7 = *v6;
        v8 = *(__int64 (**)())(*v6 + 104);
        if ( v8 == sub_EC9C80 )
          break;
        v12 = ((unsigned __int8 (__fastcall *)(__int64 *, const char *, const char *))v8)(v6, v16, v17) == 0;
        v7 = **(_QWORD **)(a1 + 8);
        if ( v12 )
          break;
        if ( **(_DWORD **)((*(__int64 (**)(void))(v7 + 40))() + 8) == 9 )
          goto LABEL_21;
      }
      v9 = (*(__int64 (**)(void))(v7 + 48))();
      v19 = 261;
      v18[0] = v16;
      v18[1] = v17;
      v10 = sub_E6C460(v9, v18);
      v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v11 + 296LL))(v11, v10, v4);
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
        goto LABEL_21;
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
        break;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    }
    v15 = *(_QWORD *)(a1 + 8);
    v18[0] = "expected comma";
    v19 = 259;
    return sub_ECE0E0(v15, v18, 0, 0);
  }
}
