// Function: sub_EC5340
// Address: 0xec5340
//
__int64 __fastcall sub_EC5340(__int64 a1)
{
  __int64 v1; // rax
  __int64 v3; // rdi
  const char *v4; // [rsp+0h] [rbp-40h] BYREF
  char v5; // [rsp+20h] [rbp-20h]
  char v6; // [rsp+21h] [rbp-1Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v1 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v1 + 224LL))(v1, 1);
    return 0;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 8);
    v6 = 1;
    v4 = "unexpected token in '.subsections_via_symbols' directive";
    v5 = 3;
    return sub_ECE0E0(v3, &v4, 0, 0);
  }
}
