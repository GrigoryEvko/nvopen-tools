// Function: sub_3901FB0
// Address: 0x3901fb0
//
__int64 __fastcall sub_3901FB0(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r9
  __int64 v3; // rax
  __int64 v5; // rdi
  const char *v6; // [rsp+0h] [rbp-30h] BYREF
  char v7; // [rsp+10h] [rbp-20h]
  char v8; // [rsp+11h] [rbp-1Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v3 + 192LL))(v3, 1);
    return 0;
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 8);
    v8 = 1;
    v6 = "unexpected token in '.subsections_via_symbols' directive";
    v7 = 3;
    return sub_3909CF0(v5, &v6, 0, 0, v1, v2);
  }
}
