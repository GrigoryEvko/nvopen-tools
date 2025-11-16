// Function: sub_8284D0
// Address: 0x8284d0
//
__int64 __fastcall sub_8284D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        int a4,
        int a5,
        int a6,
        int a7,
        __int64 *a8,
        __int64 a9)
{
  __int64 v10; // r12
  __int64 *v12; // rdi
  __int64 v14; // rax
  char v15; // dl
  __int64 v16; // r8
  __int64 v17; // rax
  int v18; // eax
  __int64 v19[7]; // [rsp+8h] [rbp-38h] BYREF

  v10 = a3;
  if ( a1 && !a3 )
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  if ( a5 )
  {
    v12 = (__int64 *)sub_6ECAE0(v10, a4, a6, a7, 3u, a8, v19);
    *(_QWORD *)(v19[0] + 56) = a2;
  }
  else
  {
    v14 = sub_6F5430(a1, a2, v10, 0, 0, 0, 0, 0, 1u, 0, (__int64)a8);
    v15 = *(_BYTE *)(v10 + 140);
    v19[0] = v14;
    v16 = v14;
    if ( v15 == 12 )
    {
      v17 = v10;
      do
      {
        v17 = *(_QWORD *)(v17 + 160);
        v15 = *(_BYTE *)(v17 + 140);
      }
      while ( v15 == 12 );
    }
    if ( v15 )
    {
      if ( a7 || (v18 = sub_6EB560(v10, (__int64)a8), v16 = v19[0], !v18) )
      {
        sub_6EB360(v16, v10, v10, a8);
        v16 = v19[0];
      }
    }
    v12 = (__int64 *)sub_6EC670(v10, v16, a4, a6);
  }
  sub_6E7170(v12, a9);
  return sub_6E26D0(2, a9);
}
