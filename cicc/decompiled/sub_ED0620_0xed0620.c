// Function: sub_ED0620
// Address: 0xed0620
//
void __fastcall sub_ED0620(__int64 a1, const void *a2, size_t a3, const void *a4, size_t a5)
{
  const char *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // [rsp+8h] [rbp-48h]
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = sub_BD5D20(a1);
  if ( (v9 != a5 || a5 && memcmp(v8, a4, a5)) && !sub_B91CC0(a1, a2, a3) )
  {
    v11 = (__int64 *)sub_BD5C60(a1);
    v12[0] = sub_B9B140(v11, a4, a5);
    v10 = sub_B9C770(v11, v12, (__int64 *)1, 0, 1);
    sub_B99460(a1, a2, a3, v10);
  }
}
