// Function: sub_2252230
// Address: 0x2252230
//
__int64 __fastcall sub_2252230(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  unsigned int v6; // r8d
  __int64 v8; // [rsp+0h] [rbp-28h] BYREF
  __int64 v9; // [rsp+8h] [rbp-20h]
  __int64 v10; // [rsp+10h] [rbp-18h]

  v4 = *a3;
  v9 = 0x1000000000LL;
  v5 = *a1;
  v8 = 0;
  v10 = 0;
  (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64 *))(v5 + 48))(a1, a2, v4, &v8);
  v6 = 0;
  if ( (v9 & 6) == 6 )
  {
    v6 = 1;
    *a3 = v8;
  }
  return v6;
}
