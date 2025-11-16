// Function: sub_23058C0
// Address: 0x23058c0
//
__int64 *__fastcall sub_23058C0(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  _QWORD v9[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v10; // [rsp+20h] [rbp-30h]

  v10 = 260;
  v9[0] = a2;
  v6 = sub_22077B0(0x40u);
  v7 = v6;
  if ( v6 )
    sub_C63EB0(v6, (__int64)v9, a3, a4);
  *a1 = v7 | 1;
  return a1;
}
