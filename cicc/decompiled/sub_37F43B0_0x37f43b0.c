// Function: sub_37F43B0
// Address: 0x37f43b0
//
bool __fastcall sub_37F43B0(__int64 a1, int a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 (*v5)(); // rcx
  __int64 (*v6)(); // rax
  int v8; // [rsp+8h] [rbp-28h] BYREF
  _DWORD v9[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *a3;
  v8 = 0;
  v9[0] = 0;
  v5 = *(__int64 (**)())(v4 + 120);
  if ( v5 != sub_2F4C0B0 )
  {
    if ( ((unsigned int (__fastcall *)(__int64 *, __int64, int *))v5)(a3, a1, &v8) )
      return v8 == a2;
    v4 = *a3;
  }
  v6 = *(__int64 (**)())(v4 + 152);
  if ( v6 == sub_2FCE890 || !((unsigned __int8 (__fastcall *)(__int64 *, __int64, int *, _DWORD *))v6)(a3, a1, &v8, v9) )
    return 0;
  return v8 == a2;
}
