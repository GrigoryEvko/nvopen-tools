// Function: sub_EC2E30
// Address: 0xec2e30
//
__int64 __fastcall sub_EC2E30(__int64 a1)
{
  __int64 v1; // rax
  char v2; // r8
  __int64 result; // rax
  __int64 v4; // rdi
  const char *v5; // [rsp+0h] [rbp-40h] BYREF
  char v6; // [rsp+20h] [rbp-20h]
  char v7; // [rsp+21h] [rbp-1Fh]

  v1 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v1 + 168LL))(v1);
  result = 0;
  if ( !v2 )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v7 = 1;
    v5 = ".popsection without corresponding .pushsection";
    v6 = 3;
    return sub_ECE0E0(v4, &v5, 0, 0);
  }
  return result;
}
