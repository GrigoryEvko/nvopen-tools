// Function: sub_766120
// Address: 0x766120
//
void __fastcall sub_766120(__int64 a1)
{
  __int64 *v1; // r13
  bool v2; // bl
  _QWORD *i; // r14
  __int64 *v4; // r13
  _DWORD v5[9]; // [rsp+Ch] [rbp-24h] BYREF

  v1 = *(__int64 **)(a1 + 256);
  v2 = *(_BYTE *)(a1 + 28) == 2 || *(_BYTE *)(a1 + 28) == 17;
  do
  {
    v5[0] = 0;
    sub_765EA0(v1, v2, v5);
  }
  while ( v5[0] );
  for ( i = *(_QWORD **)(a1 + 264); i; i = (_QWORD *)*i )
  {
    v4 = (__int64 *)i[1];
    do
    {
      v5[0] = 0;
      sub_765EA0(v4, v2, v5);
    }
    while ( v5[0] );
  }
}
