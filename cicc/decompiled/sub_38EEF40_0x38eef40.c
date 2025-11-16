// Function: sub_38EEF40
// Address: 0x38eef40
//
__int64 __fastcall sub_38EEF40(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  const char *v10; // rax
  unsigned __int8 v11; // [rsp+17h] [rbp-59h] BYREF
  __int64 v12; // [rsp+18h] [rbp-58h] BYREF
  __int64 v13; // [rsp+20h] [rbp-50h] BYREF
  __int64 v14; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v15[2]; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v16; // [rsp+40h] [rbp-30h]

  v1 = sub_3909460(a1);
  v2 = sub_39092A0(v1);
  if ( (unsigned __int8)sub_38E31C0(a1, &v12, (__int64)".cv_loc", 7)
    || (unsigned __int8)sub_38E3370(a1, &v13, (__int64)".cv_loc", 7) )
  {
    return 1;
  }
  LODWORD(v4) = 0;
  LODWORD(v5) = 0;
  if ( **(_DWORD **)(a1 + 152) == 4 )
  {
    v6 = sub_3909460(a1);
    if ( *(_DWORD *)(v6 + 32) <= 0x40u )
      v4 = *(_QWORD *)(v6 + 24);
    else
      v4 = **(_QWORD **)(v6 + 24);
    if ( v4 < 0 )
    {
      BYTE1(v16) = 1;
      v10 = "line number less than zero in '.cv_loc' directive";
    }
    else
    {
      sub_38EB180(a1);
      if ( **(_DWORD **)(a1 + 152) != 4 )
      {
        LODWORD(v5) = 0;
        goto LABEL_6;
      }
      v9 = sub_3909460(a1);
      if ( *(_DWORD *)(v9 + 32) <= 0x40u )
        v5 = *(_QWORD *)(v9 + 24);
      else
        v5 = **(_QWORD **)(v9 + 24);
      if ( v5 >= 0 )
      {
        sub_38EB180(a1);
        goto LABEL_6;
      }
      BYTE1(v16) = 1;
      v10 = "column position less than zero in '.cv_loc' directive";
    }
    v15[0] = v10;
    LOBYTE(v16) = 3;
    return sub_3909CF0(a1, v15, 0, 0, v7, v8);
  }
LABEL_6:
  v15[1] = &v11;
  v11 = 0;
  v14 = 0;
  v15[0] = a1;
  v16 = &v14;
  result = sub_3909F10(a1, sub_38F38E0, v15, 0);
  if ( !(_BYTE)result )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, bool, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 328) + 624LL))(
      *(_QWORD *)(a1 + 328),
      (unsigned int)v12,
      (unsigned int)v13,
      (unsigned int)v4,
      (unsigned int)v5,
      v11,
      v14 != 0,
      0,
      0,
      v2);
    return 0;
  }
  return result;
}
