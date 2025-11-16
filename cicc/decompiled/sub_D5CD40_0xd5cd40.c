// Function: sub_D5CD40
// Address: 0xd5cd40
//
__int64 __fastcall sub_D5CD40(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __m128i v4; // [rsp+0h] [rbp-30h] BYREF
  int v5; // [rsp+10h] [rbp-20h]
  char v6; // [rsp+18h] [rbp-18h]

  v2 = sub_D5BAA0((unsigned __int8 *)a1);
  if ( v2
    && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v2 + 24) + 16LL) + 8LL) == 14
    && (sub_D5BC90(&v4, v2, 7u, a2), v5 >= 0)
    && v6 )
  {
    return *(_QWORD *)(a1 + 32 * (v5 - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  }
  else
  {
    return sub_B494D0(a1, 1);
  }
}
