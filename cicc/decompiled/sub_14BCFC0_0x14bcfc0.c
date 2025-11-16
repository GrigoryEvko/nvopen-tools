// Function: sub_14BCFC0
// Address: 0x14bcfc0
//
__int64 __fastcall sub_14BCFC0(__int64 a1, __int64 *a2, unsigned int a3, __int64 *a4)
{
  unsigned int v6; // ebx
  __int64 v8; // [rsp+8h] [rbp-38h]

  v8 = *a2;
  v6 = sub_16431D0(*a2);
  if ( !v6 )
    v6 = sub_15A95F0(*a4, v8);
  *(_DWORD *)(a1 + 8) = v6;
  if ( v6 > 0x40 )
  {
    sub_16A4EF0(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = v6;
    sub_16A4EF0(a1 + 16, 0, 0);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = 0;
  }
  sub_14B86A0(a2, a1, a3, a4);
  return a1;
}
