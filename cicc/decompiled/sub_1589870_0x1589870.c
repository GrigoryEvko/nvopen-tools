// Function: sub_1589870
// Address: 0x1589870
//
__int64 __fastcall sub_1589870(__int64 a1, __int64 *a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-18h]

  v2 = *((_DWORD *)a2 + 2);
  v3 = *a2;
  *((_DWORD *)a2 + 2) = 0;
  *(_DWORD *)(a1 + 8) = v2;
  *(_QWORD *)a1 = v3;
  v6 = v2;
  if ( v2 > 0x40 )
    sub_16A4FD0(&v5, a1);
  else
    v5 = v3;
  sub_16A7490(&v5, 1);
  *(_DWORD *)(a1 + 24) = v6;
  result = v5;
  *(_QWORD *)(a1 + 16) = v5;
  return result;
}
