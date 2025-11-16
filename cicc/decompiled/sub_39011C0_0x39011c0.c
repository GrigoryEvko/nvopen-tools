// Function: sub_39011C0
// Address: 0x39011c0
//
__int64 __fastcall sub_39011C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned int v7; // r12d
  __int64 v8; // rax
  const char *v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // [rsp+4h] [rbp-4Ch] BYREF
  unsigned int v13; // [rsp+8h] [rbp-48h] BYREF
  const char *v14; // [rsp+10h] [rbp-40h] BYREF
  char v15; // [rsp+20h] [rbp-30h]
  char v16; // [rsp+21h] [rbp-2Fh]

  v12 = 0;
  if ( !(unsigned __int8)sub_38FF7C0(a1, (int *)&v12) )
  {
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
    {
      v16 = 1;
      v10 = "you must specify a stack pointer offset";
LABEL_7:
      v11 = *(_QWORD *)(a1 + 8);
      v14 = v10;
      v15 = 3;
      return (unsigned int)sub_3909CF0(v11, &v14, 0, 0, v5, v6);
    }
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v7 = (*(__int64 (__fastcall **)(_QWORD, unsigned int *))(**(_QWORD **)(a1 + 8) + 200LL))(*(_QWORD *)(a1 + 8), &v13);
    if ( !(_BYTE)v7 )
    {
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
      {
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
        v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
        (*(void (__fastcall **)(__int64, _QWORD, _QWORD, __int64))(*(_QWORD *)v8 + 912LL))(v8, v12, v13, a4);
        return v7;
      }
      v16 = 1;
      v10 = "unexpected token in directive";
      goto LABEL_7;
    }
  }
  return 1;
}
