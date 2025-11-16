// Function: sub_24F30B0
// Address: 0x24f30b0
//
__int64 __fastcall sub_24F30B0(__int64 ***a1, __int64 **a2)
{
  __int64 *v3; // rdi
  __int64 v4; // rax
  __int64 **v5; // rdi
  __int64 *v6; // rax
  unsigned __int64 v7; // rax
  __int64 **v8; // rdi
  __int64 result; // rax
  __int64 v10[3]; // [rsp+8h] [rbp-18h] BYREF

  *a1 = a2;
  v3 = *a2;
  a1[1] = (__int64 **)*a2;
  v4 = sub_BCE3C0(v3, 0);
  v5 = a1[1];
  a1[2] = (__int64 **)v4;
  v10[0] = v4;
  v6 = (__int64 *)sub_BCB120(v5);
  v7 = sub_BCF480(v6, v10, 1, 0);
  v8 = a1[2];
  a1[3] = (__int64 **)v7;
  result = sub_AC9EC0(v8);
  a1[4] = (__int64 **)result;
  return result;
}
