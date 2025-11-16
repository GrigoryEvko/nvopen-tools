// Function: sub_EC52A0
// Address: 0xec52a0
//
__int64 __fastcall sub_EC52A0(__int64 a1)
{
  __int64 v1; // rdi
  void (*v2)(); // rdx
  __int64 result; // rax
  __int64 v4; // rdi
  const char *v5; // [rsp+10h] [rbp-40h] BYREF
  char v6; // [rsp+30h] [rbp-20h]
  char v7; // [rsp+31h] [rbp-1Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v1 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v2 = *(void (**)())(*(_QWORD *)v1 + 240LL);
    result = 0;
    if ( v2 != nullsub_101 )
    {
      ((void (__fastcall *)(__int64, __int64))v2)(v1, 4);
      return 0;
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 8);
    v7 = 1;
    v5 = "unexpected token in '.end_data_region' directive";
    v6 = 3;
    return sub_ECE0E0(v4, &v5, 0, 0);
  }
  return result;
}
