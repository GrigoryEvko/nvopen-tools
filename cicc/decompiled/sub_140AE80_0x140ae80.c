// Function: sub_140AE80
// Address: 0x140ae80
//
__int64 __fastcall sub_140AE80(__int64 a1, __int64 *a2)
{
  unsigned int v3; // esi
  __int64 v4; // rax
  unsigned int v5; // ebx
  __int64 v7; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-38h]

  v3 = *((_DWORD *)a2 + 6);
  v4 = a2[2];
  if ( v3 > 0x40 )
    v4 = *(_QWORD *)(v4 + 8LL * ((v3 - 1) >> 6));
  v5 = *((_DWORD *)a2 + 2);
  if ( (v4 & (1LL << ((unsigned __int8)v3 - 1))) != 0 || (int)sub_16A9900(a2, a2 + 2) < 0 )
  {
    *(_DWORD *)(a1 + 8) = v5;
    if ( v5 <= 0x40 )
      *(_QWORD *)a1 = 0;
    else
      sub_16A4EF0(a1, 0, 0);
  }
  else
  {
    v8 = v5;
    if ( v5 > 0x40 )
      sub_16A4FD0(&v7, a2);
    else
      v7 = *a2;
    sub_16A7590(&v7, a2 + 2);
    *(_DWORD *)(a1 + 8) = v8;
    *(_QWORD *)a1 = v7;
  }
  return a1;
}
