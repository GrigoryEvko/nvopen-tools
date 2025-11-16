// Function: sub_394ACF0
// Address: 0x394acf0
//
__int64 __fastcall sub_394ACF0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r13
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // r14
  int v8; // r13d
  char v9; // al

  v4 = *(_QWORD *)(a1 + 768);
  *(_QWORD *)(a1 + 760) = a2;
  if ( v4 )
  {
    j___libc_free_0(*(_QWORD *)(v4 + 8));
    j_j___libc_free_0(v4);
  }
  v5 = sub_22077B0(0x20u);
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_DWORD *)(v5 + 24) = 0;
  }
  *(_QWORD *)(a1 + 768) = v5;
  v6 = sub_17004C0(a3);
  v7 = *(_QWORD *)(a1 + 760);
  v8 = v6;
  v9 = sub_17004A0(a3);
  return sub_38D34B0(a1 + 8, a3 + 472, v9, v7, v8 == 3);
}
