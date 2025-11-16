// Function: sub_2E09700
// Address: 0x2e09700
//
__int64 __fastcall sub_2E09700(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v4; // rax
  __int64 v5; // [rsp+0h] [rbp-40h] BYREF
  __int64 v6; // [rsp+8h] [rbp-38h]
  int v7; // [rsp+10h] [rbp-30h]
  __int16 v8; // [rsp+14h] [rbp-2Ch]
  char v9; // [rsp+16h] [rbp-2Ah]

  v2 = *(__int64 **)(a1 + 8);
  if ( v2[1] )
  {
    v5 = v2[1];
    v6 = 0;
    v7 = 16;
    v8 = 257;
    v9 = 0;
    sub_CB6AF0(a2, (__int64)&v5);
    v4 = *v2;
    v6 = 0;
    v5 = v4;
  }
  else
  {
    v5 = *v2;
    v6 = 0;
  }
  v7 = 16;
  v8 = 257;
  v9 = 0;
  return sub_CB6AF0(a2, (__int64)&v5);
}
