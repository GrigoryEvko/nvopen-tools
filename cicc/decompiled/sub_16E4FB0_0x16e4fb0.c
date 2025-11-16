// Function: sub_16E4FB0
// Address: 0x16e4fb0
//
__int64 __fastcall sub_16E4FB0(__int64 a1, __int64 *a2)
{
  int v2; // r13d
  __int64 result; // rax
  int v4; // ebx
  __int64 v5; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v6[10]; // [rsp+10h] [rbp-50h] BYREF

  if ( *(_DWORD *)(a1 + 40) )
    sub_16E4E00(a1);
  v2 = 1;
  sub_16E4B40(a1, " |", 2u);
  sub_16E4DB0(a1);
  if ( *(_DWORD *)(a1 + 40) )
    v2 = *(_DWORD *)(a1 + 40);
  sub_16C2450(&v5, *a2, a2[1], (__int64)byte_3F871B3, 0);
  for ( result = sub_16F4AE0(v6, v5, 0, 0); v6[0]; result = sub_16F48F0(v6) )
  {
    v4 = 0;
    do
    {
      ++v4;
      sub_16E4B40(a1, "  ", 2u);
    }
    while ( v4 != v2 );
    sub_16E4B40(a1, (const char *)v6[2], v6[3]);
    sub_16E4DB0(a1);
  }
  if ( v5 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  return result;
}
