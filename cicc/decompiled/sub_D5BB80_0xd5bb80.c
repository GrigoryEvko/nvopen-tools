// Function: sub_D5BB80
// Address: 0xd5bb80
//
__int64 __fastcall sub_D5BB80(unsigned __int8 *a1)
{
  __int64 v1; // r13
  int v2; // eax
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = 0;
  v2 = *a1;
  if ( (unsigned __int8)v2 <= 0x1Cu )
    return v1;
  v3 = (unsigned int)(v2 - 34);
  if ( (unsigned __int8)v3 > 0x33u )
    return v1;
  v4 = 0x8000000000041LL;
  if ( !_bittest64(&v4, v3) )
    return v1;
  v7[0] = *((_QWORD *)a1 + 9);
  v5 = sub_A747F0(v7, -1, 87);
  if ( v5 )
  {
    v7[0] = v5;
    return sub_A71B80(v7);
  }
  v7[0] = sub_B495C0((__int64)a1, 87);
  if ( v7[0] )
    return sub_A71B80(v7);
  return 0;
}
