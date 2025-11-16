// Function: sub_14EDF60
// Address: 0x14edf60
//
__int64 __fastcall sub_14EDF60(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rax

  v7 = sub_1648A60(88, 1);
  if ( v7 )
  {
    v8 = sub_15FB2A0(*a1, a2, a3);
    sub_15F1EA0(v7, v8, 62, v7 - 24, 1, a5);
    if ( *(_QWORD *)(v7 - 24) )
    {
      v9 = *(_QWORD *)(v7 - 16);
      v10 = *(_QWORD *)(v7 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v10 = v9;
      if ( v9 )
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
    }
    *(_QWORD *)(v7 - 24) = a1;
    v11 = a1[1];
    *(_QWORD *)(v7 - 16) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = (v7 - 16) | *(_QWORD *)(v11 + 16) & 3LL;
    *(_QWORD *)(v7 - 8) = (unsigned __int64)(a1 + 1) | *(_QWORD *)(v7 - 8) & 3LL;
    a1[1] = v7 - 24;
    *(_QWORD *)(v7 + 56) = v7 + 72;
    *(_QWORD *)(v7 + 64) = 0x400000000LL;
    sub_15FB110(v7, a2, a3, a4);
  }
  return v7;
}
