// Function: sub_2353B90
// Address: 0x2353b90
//
unsigned __int64 __fastcall sub_2353B90(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  char v3; // r13
  __int64 v4; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *a2;
  v3 = *((_BYTE *)a2 + 8);
  v4 = sub_22077B0(0x18u);
  if ( v4 )
  {
    *(_QWORD *)(v4 + 8) = v2;
    *(_BYTE *)(v4 + 16) = v3;
    *(_QWORD *)v4 = &unk_4A10738;
  }
  v6[0] = v4;
  result = sub_2353900(a1, v6);
  if ( v6[0] )
    return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v6[0] + 8LL))(v6[0]);
  return result;
}
