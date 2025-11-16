// Function: sub_38FF7C0
// Address: 0x38ff7c0
//
__int64 __fastcall sub_38FF7C0(__int64 a1, int *a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // r12d
  int v9; // eax
  __int64 v11; // r14
  __int64 v12; // rax
  const char *v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // [rsp+Ch] [rbp-54h] BYREF
  __int64 v16; // [rsp+10h] [rbp-50h] BYREF
  __int64 v17; // [rsp+18h] [rbp-48h] BYREF
  const char *v18; // [rsp+20h] [rbp-40h] BYREF
  char v19; // [rsp+30h] [rbp-30h]
  char v20; // [rsp+31h] [rbp-2Fh]

  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v4 = sub_3909290(v3);
  v5 = *(_QWORD *)(a1 + 8);
  v16 = v4;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 40LL))(v5);
  v7 = *(_QWORD *)(a1 + 8);
  if ( **(_DWORD **)(v6 + 8) == 36 )
  {
    v11 = *(_QWORD *)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 48LL))(v7) + 24);
    v12 = *(_QWORD *)(a1 + 8);
    v17 = 0;
    v8 = (*(__int64 (__fastcall **)(_QWORD, unsigned int *, __int64 *, __int64 *))(**(_QWORD **)(v12 + 8) + 32LL))(
           *(_QWORD *)(v12 + 8),
           &v15,
           &v16,
           &v17);
    if ( (_BYTE)v8 )
      return v8;
    v9 = sub_38D7240(v11, v15);
    if ( v9 >= 0 )
      goto LABEL_4;
    v20 = 1;
    v13 = "register can't be represented in SEH unwind info";
  }
  else
  {
    v8 = (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v7 + 200LL))(v7, &v17);
    if ( (_BYTE)v8 )
      return v8;
    v9 = v17;
    if ( v17 <= 15 )
    {
LABEL_4:
      *a2 = v9;
      return v8;
    }
    v20 = 1;
    v13 = "register number is too high";
  }
  v14 = *(_QWORD *)(a1 + 8);
  v18 = v13;
  v19 = 3;
  return (unsigned int)sub_3909790(v14, v16, &v18, 0, 0);
}
