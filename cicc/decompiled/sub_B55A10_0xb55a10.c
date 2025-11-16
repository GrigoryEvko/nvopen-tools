// Function: sub_B55A10
// Address: 0xb55a10
//
__int64 __fastcall sub_B55A10(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  char v8[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v9; // [rsp+20h] [rbp-30h]

  v1 = *(_QWORD *)(a1 - 32);
  v2 = *(_QWORD *)(a1 + 8);
  v9 = 257;
  v3 = sub_BD2C40(72, unk_3F10A14);
  v4 = v3;
  if ( v3 )
  {
    sub_B44260(v3, v2, 60, 1u, 0, 0);
    if ( *(_QWORD *)(v4 - 32) )
    {
      v5 = *(_QWORD *)(v4 - 24);
      **(_QWORD **)(v4 - 16) = v5;
      if ( v5 )
        *(_QWORD *)(v5 + 16) = *(_QWORD *)(v4 - 16);
    }
    *(_QWORD *)(v4 - 32) = v1;
    if ( v1 )
    {
      v6 = *(_QWORD *)(v1 + 16);
      *(_QWORD *)(v4 - 24) = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = v4 - 24;
      *(_QWORD *)(v4 - 16) = v1 + 16;
      *(_QWORD *)(v1 + 16) = v4 - 32;
    }
    sub_BD6B50(v4, v8);
  }
  return v4;
}
