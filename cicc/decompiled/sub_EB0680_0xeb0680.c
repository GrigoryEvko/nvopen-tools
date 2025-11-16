// Function: sub_EB0680
// Address: 0xeb0680
//
__int64 __fastcall sub_EB0680(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  const char *v8; // rax
  unsigned __int8 v9; // [rsp+17h] [rbp-69h] BYREF
  __int64 v10; // [rsp+18h] [rbp-68h] BYREF
  __int64 v11; // [rsp+20h] [rbp-60h] BYREF
  __int64 v12; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v13[4]; // [rsp+30h] [rbp-50h] BYREF
  char v14; // [rsp+50h] [rbp-30h]
  char v15; // [rsp+51h] [rbp-2Fh]

  v1 = sub_ECD7B0(a1);
  v2 = sub_ECD6A0(v1);
  if ( (unsigned __int8)sub_EA2660(a1, &v10) || (unsigned __int8)sub_EA3FF0(a1, &v11, (__int64)".cv_loc", 7) )
    return 1;
  LODWORD(v4) = 0;
  LODWORD(v5) = 0;
  if ( **(_DWORD **)(a1 + 48) == 4 )
  {
    v6 = sub_ECD7B0(a1);
    if ( *(_DWORD *)(v6 + 32) <= 0x40u )
      v4 = *(_QWORD *)(v6 + 24);
    else
      v4 = **(_QWORD **)(v6 + 24);
    if ( v4 < 0 )
    {
      v15 = 1;
      v8 = "line number less than zero in '.cv_loc' directive";
    }
    else
    {
      sub_EABFE0(a1);
      if ( **(_DWORD **)(a1 + 48) != 4 )
      {
        LODWORD(v5) = 0;
        goto LABEL_6;
      }
      v7 = sub_ECD7B0(a1);
      if ( *(_DWORD *)(v7 + 32) <= 0x40u )
        v5 = *(_QWORD *)(v7 + 24);
      else
        v5 = **(_QWORD **)(v7 + 24);
      if ( v5 >= 0 )
      {
        sub_EABFE0(a1);
        goto LABEL_6;
      }
      v15 = 1;
      v8 = "column position less than zero in '.cv_loc' directive";
    }
    v13[0] = v8;
    v14 = 3;
    return sub_ECE0E0(a1, v13, 0, 0);
  }
LABEL_6:
  v13[1] = &v9;
  v9 = 0;
  v12 = 0;
  v13[0] = a1;
  v13[2] = &v12;
  result = sub_ECE300(a1, sub_EB8900, v13, 0);
  if ( !(_BYTE)result )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, bool, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 232) + 736LL))(
      *(_QWORD *)(a1 + 232),
      (unsigned int)v10,
      (unsigned int)v11,
      (unsigned int)v4,
      (unsigned int)v5,
      v9,
      v12 != 0,
      0,
      0,
      v2);
    return 0;
  }
  return result;
}
