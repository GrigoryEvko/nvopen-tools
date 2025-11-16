// Function: sub_EAD990
// Address: 0xead990
//
__int64 __fastcall sub_EAD990(__int64 *a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // r13
  unsigned int v5; // eax
  unsigned int v6; // r12d
  __int64 v7; // rdi
  int v8; // edx
  int v9; // eax
  signed __int64 v10; // rsi
  __int64 v11; // rax
  signed __int64 v12; // rax
  unsigned __int8 *v14; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v15[4]; // [rsp+10h] [rbp-50h] BYREF
  char v16; // [rsp+30h] [rbp-30h]
  char v17; // [rsp+31h] [rbp-2Fh]

  v2 = sub_ECD690(*a1 + 40);
  v3 = *a1;
  v4 = v2;
  if ( !*(_BYTE *)(*a1 + 869) )
  {
    if ( (unsigned __int8)sub_EA2540(v3) )
      return 1;
    v3 = *a1;
  }
  v15[0] = 0;
  LOBYTE(v5) = sub_EAC4D0(v3, (__int64 *)&v14, (__int64)v15);
  v6 = v5;
  if ( !(_BYTE)v5 )
  {
    v7 = *a1;
    v8 = *(_DWORD *)a1[1];
    if ( *v14 != 1 )
    {
      sub_E9A5B0(*(_QWORD *)(v7 + 232), v14);
      return v6;
    }
    v9 = 8 * v8;
    v10 = *((_QWORD *)v14 + 2);
    if ( (unsigned int)(8 * v8) <= 0x3F )
    {
      if ( !v9 )
      {
        if ( !v10 )
          goto LABEL_12;
        if ( v10 >= 0 )
        {
          v12 = 0;
LABEL_11:
          if ( v10 <= v12 )
            goto LABEL_12;
        }
LABEL_18:
        v17 = 1;
        v15[0] = "out of range literal value";
        v16 = 3;
        return (unsigned int)sub_ECDA70(v7, v4, v15, 0, 0);
      }
      if ( v10 > 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) )
      {
        v11 = 1LL << ((unsigned __int8)v9 - 1);
        if ( v10 >= -v11 )
        {
          v12 = v11 - 1;
          goto LABEL_11;
        }
        goto LABEL_18;
      }
    }
LABEL_12:
    (*(void (__fastcall **)(_QWORD, signed __int64))(**(_QWORD **)(v7 + 232) + 536LL))(*(_QWORD *)(v7 + 232), v10);
    return v6;
  }
  return 1;
}
