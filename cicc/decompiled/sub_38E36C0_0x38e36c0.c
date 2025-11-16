// Function: sub_38E36C0
// Address: 0x38e36c0
//
__int64 __fastcall sub_38E36C0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  const char *v7; // [rsp+0h] [rbp-30h] BYREF
  char v8; // [rsp+10h] [rbp-20h]
  char v9; // [rsp+11h] [rbp-1Fh]

  v2 = *(_QWORD *)(a1 + 328);
  v3 = *(unsigned int *)(v2 + 120);
  if ( (_DWORD)v3 && *(_QWORD *)(*(_QWORD *)(v2 + 112) + 32 * v3 - 32) )
    return 0;
  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v2 + 168LL))(v2, 0);
  v9 = 1;
  v7 = "expected section directive before assembly directive";
  v8 = 3;
  v4 = sub_3909460(a1);
  v5 = sub_39092A0(v4);
  return sub_3909790(a1, v5, &v7, 0, 0);
}
