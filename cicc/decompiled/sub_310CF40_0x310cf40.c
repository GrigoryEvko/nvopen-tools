// Function: sub_310CF40
// Address: 0x310cf40
//
__int64 __fastcall sub_310CF40(__int64 *a1)
{
  __int64 result; // rax
  __int64 *v2; // rbx
  __int64 *i; // r13
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int16 v7; // [rsp+0h] [rbp-50h] BYREF
  __int64 v8; // [rsp+8h] [rbp-48h]

  result = *a1;
  v2 = *(__int64 **)(*a1 + 40);
  for ( i = *(__int64 **)(*a1 + 48); i != v2; result = sub_C6BC50(&v7) )
  {
    v5 = *v2;
    v6 = a1[1];
    v7 = 3;
    ++v2;
    v8 = v5;
    sub_C6C710(v6, &v7, 3);
  }
  return result;
}
