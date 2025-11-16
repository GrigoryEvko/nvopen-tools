// Function: sub_23CE520
// Address: 0x23ce520
//
__int64 *__fastcall sub_23CE520(__int64 *a1)
{
  unsigned int v1; // eax
  __int64 v2; // rdx
  __int64 v3; // r13
  unsigned int v4; // r14d
  __int64 v5; // rax
  __int64 v6; // rbx
  const char *v8; // [rsp+0h] [rbp-50h] BYREF
  char v9; // [rsp+20h] [rbp-30h]
  char v10; // [rsp+21h] [rbp-2Fh]

  v1 = sub_C63BB0();
  v10 = 1;
  v8 = "buildCodeGenPipeline is not overridden";
  v3 = v2;
  v4 = v1;
  v9 = 3;
  v5 = sub_22077B0(0x40u);
  v6 = v5;
  if ( v5 )
    sub_C63EB0(v5, (__int64)&v8, v4, v3);
  *a1 = v6 | 1;
  return a1;
}
