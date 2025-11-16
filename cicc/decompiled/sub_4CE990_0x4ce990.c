// Function: sub_4CE990
// Address: 0x4ce990
//
void __fastcall sub_4CE990(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  const char *v4; // [rsp-38h] [rbp-38h] BYREF
  char v5; // [rsp-28h] [rbp-28h]
  char v6; // [rsp-27h] [rbp-27h]

  if ( *a1 )
  {
    v3 = sub_16E8CB0();
    v6 = 1;
    v4 = "cl::location(x) specified more than once!";
    v5 = 3;
    sub_16B1F90(a2, &v4, 0, 0, v3);
  }
  else
  {
    *a1 = a3;
  }
}
