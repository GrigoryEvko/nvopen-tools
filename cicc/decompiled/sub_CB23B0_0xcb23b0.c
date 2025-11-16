// Function: sub_CB23B0
// Address: 0xcb23b0
//
__int64 __fastcall sub_CB23B0(__int64 a1, __int64 *a2)
{
  int v2; // r13d
  __int64 result; // rax
  int v4; // ebx
  __int64 v5; // [rsp+8h] [rbp-78h] BYREF
  _BYTE v6[32]; // [rsp+10h] [rbp-70h] BYREF
  char v7; // [rsp+30h] [rbp-50h]
  const void *v8; // [rsp+40h] [rbp-40h]
  size_t v9; // [rsp+48h] [rbp-38h]

  if ( *(_DWORD *)(a1 + 40) )
    sub_CB20A0(a1, 0);
  v2 = 1;
  sub_CB1B10(a1, " |", 2u);
  sub_CB1E40(a1);
  if ( *(_DWORD *)(a1 + 40) )
    v2 = *(_DWORD *)(a1 + 40);
  sub_C7DA90(&v5, *a2, a2[1], byte_3F871B3, 0, 0);
  for ( result = sub_C7C840((__int64)v6, v5, 0, 0); v7; result = sub_C7C5C0((__int64)v6) )
  {
    v4 = 0;
    do
    {
      ++v4;
      sub_CB1B10(a1, "  ", 2u);
    }
    while ( v2 != v4 );
    sub_CB1B10(a1, v8, v9);
    sub_CB1E40(a1);
  }
  if ( v5 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  return result;
}
