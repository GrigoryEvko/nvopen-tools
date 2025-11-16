// Function: sub_3904600
// Address: 0x3904600
//
__int64 __fastcall sub_3904600(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdi
  unsigned int v13; // r10d
  unsigned int v15; // r15d
  const char *v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  void (*v21)(); // rax
  __int64 v22; // rdi
  unsigned int v23; // [rsp+14h] [rbp-6Ch] BYREF
  unsigned int v24; // [rsp+18h] [rbp-68h] BYREF
  unsigned int v25; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v26; // [rsp+20h] [rbp-60h] BYREF
  __int64 v27; // [rsp+28h] [rbp-58h]
  _QWORD v28[2]; // [rsp+30h] [rbp-50h] BYREF
  char v29; // [rsp+40h] [rbp-40h]
  char v30; // [rsp+41h] [rbp-3Fh]

  v7 = *(_QWORD *)(a1 + 8);
  v26 = 0;
  v27 = 0;
  v8 = sub_3909460(v7);
  v9 = sub_39092A0(v8);
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 144LL))(*(_QWORD *)(a1 + 8), &v26) )
  {
    v30 = 1;
    v16 = "platform name expected";
LABEL_12:
    v17 = *(_QWORD *)(a1 + 8);
    v28[0] = v16;
    v29 = 3;
    return (unsigned int)sub_3909CF0(v17, v28, 0, 0, v10, v11);
  }
  if ( v27 != 5 )
  {
    if ( v27 == 3 )
    {
      if ( *(_WORD *)v26 == 28521 && *(_BYTE *)(v26 + 2) == 115 )
      {
        v15 = 2;
        goto LABEL_19;
      }
    }
    else if ( v27 == 4 )
    {
      if ( *(_DWORD *)v26 == 1936684660 )
      {
        v15 = 3;
        goto LABEL_19;
      }
    }
    else if ( v27 == 7 && *(_DWORD *)v26 == 1668571511 && *(_WORD *)(v26 + 4) == 28520 && *(_BYTE *)(v26 + 6) == 115 )
    {
      v15 = 4;
      goto LABEL_19;
    }
LABEL_6:
    v12 = *(_QWORD *)(a1 + 8);
    v30 = 1;
    v28[0] = "unknown platform name";
    v29 = 3;
    return (unsigned int)sub_3909790(v12, v9, v28, 0, 0);
  }
  if ( *(_DWORD *)v26 != 1868783981 || *(_BYTE *)(v26 + 4) != 115 )
    goto LABEL_6;
  v15 = 1;
LABEL_19:
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64))(**(_QWORD **)(a1 + 8) + 40LL))(
                       *(_QWORD *)(a1 + 8),
                       &v26,
                       v26)
                   + 8) != 25 )
  {
    v30 = 1;
    v16 = "version number required, comma expected";
    goto LABEL_12;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v13 = sub_3904260(a1, &v23, &v24, &v25);
  if ( !(_BYTE)v13 )
  {
    v18 = *(_QWORD *)(a1 + 8);
    v30 = 1;
    v28[0] = "unexpected token";
    v29 = 3;
    if ( (unsigned __int8)sub_3909E20(v18, 9, v28) )
    {
      v22 = *(_QWORD *)(a1 + 8);
      v30 = 1;
      v28[0] = " in '.build_version' directive";
      v29 = 3;
      return (unsigned int)sub_39094A0(v22, v28);
    }
    else
    {
      sub_39037E0(a1, a2, a3, v26, v27, a4, *(_DWORD *)&asc_452FBA0[4 * v15 - 4]);
      v19 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v13 = 0;
      v20 = v19;
      v21 = *(void (**)())(*(_QWORD *)v19 + 224LL);
      if ( v21 != nullsub_585 )
      {
        ((void (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v21)(v20, v15, v23, v24, v25);
        return 0;
      }
    }
  }
  return v13;
}
