// Function: sub_EC3020
// Address: 0xec3020
//
__int64 __fastcall sub_EC3020(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v2; // rax
  __int64 v4; // rdi
  unsigned int v5; // [rsp+8h] [rbp-48h] BYREF
  const char *v6; // [rsp+10h] [rbp-40h] BYREF
  char v7; // [rsp+30h] [rbp-20h]
  char v8; // [rsp+31h] [rbp-1Fh]

  v1 = (*(__int64 (__fastcall **)(_QWORD, unsigned int *))(**(_QWORD **)(a1 + 8) + 256LL))(*(_QWORD *)(a1 + 8), &v5);
  if ( (_BYTE)v1 )
    return v1;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v2 + 328LL))(v2, v5);
    return v1;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v6 = "unexpected token in directive";
  v7 = 3;
  return (unsigned int)sub_ECE0E0(v4, &v6, 0, 0);
}
