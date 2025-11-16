// Function: sub_384F350
// Address: 0x384f350
//
void __fastcall sub_384F350(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // [rsp+8h] [rbp-28h] BYREF
  _QWORD v4[2]; // [rsp+10h] [rbp-20h] BYREF
  __int64 v5; // [rsp+20h] [rbp-10h]
  __int64 v6; // [rsp+28h] [rbp-8h]

  v2 = *(_DWORD *)(a1 + 184);
  v4[0] = 0;
  v4[1] = -1;
  v5 = 0;
  v6 = 0;
  if ( v2 && *(_DWORD *)(a1 + 216) && sub_384F1D0(a1, a2, &v3, v4) )
    sub_384F170(a1, v5);
}
