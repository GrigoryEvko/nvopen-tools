// Function: sub_314D920
// Address: 0x314d920
//
unsigned __int64 __fastcall sub_314D920(unsigned __int64 *a1, int a2)
{
  __int64 v2; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v4[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_22077B0(0x10u);
  if ( v2 )
  {
    *(_DWORD *)(v2 + 8) = a2;
    *(_QWORD *)v2 = &unk_4A0F878;
  }
  v4[0] = v2;
  result = sub_314D790(a1, v4);
  if ( v4[0] )
    return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v4[0] + 8LL))(v4[0]);
  return result;
}
