// Function: sub_F34910
// Address: 0xf34910
//
__int64 __fastcall sub_F34910(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  bool v3; // zf
  __int64 v4; // rdi
  __int64 v6; // [rsp+0h] [rbp-10h] BYREF
  __int16 v7; // [rsp+8h] [rbp-8h]

  v2 = a1 + 24;
  v3 = a1 == 0;
  v4 = *(_QWORD *)(a1 + 40);
  if ( v3 )
    v2 = 0;
  v6 = v2;
  v7 = 0;
  return sub_F346C0(v4, (__int64)&v6, a2);
}
