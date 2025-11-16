// Function: sub_B55890
// Address: 0xb55890
//
__int64 __fastcall sub_B55890(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  char v13[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v14; // [rsp+20h] [rbp-30h]

  v1 = *(_QWORD *)(a1 - 32);
  v2 = *(_QWORD *)(a1 - 64);
  v14 = 257;
  v3 = *(_QWORD *)(a1 - 96);
  v4 = sub_BD2C40(72, 3);
  v5 = v4;
  if ( v4 )
  {
    sub_B44260(v4, *(_QWORD *)(v2 + 8), 57, 3u, 0, 0);
    if ( *(_QWORD *)(v5 - 96) )
    {
      v6 = *(_QWORD *)(v5 - 88);
      **(_QWORD **)(v5 - 80) = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = *(_QWORD *)(v5 - 80);
    }
    *(_QWORD *)(v5 - 96) = v3;
    if ( v3 )
    {
      v7 = *(_QWORD *)(v3 + 16);
      *(_QWORD *)(v5 - 88) = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = v5 - 88;
      *(_QWORD *)(v5 - 80) = v3 + 16;
      *(_QWORD *)(v3 + 16) = v5 - 96;
    }
    if ( *(_QWORD *)(v5 - 64) )
    {
      v8 = *(_QWORD *)(v5 - 56);
      **(_QWORD **)(v5 - 48) = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 16) = *(_QWORD *)(v5 - 48);
    }
    *(_QWORD *)(v5 - 64) = v2;
    v9 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v5 - 56) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = v5 - 56;
    *(_QWORD *)(v5 - 48) = v2 + 16;
    *(_QWORD *)(v2 + 16) = v5 - 64;
    if ( *(_QWORD *)(v5 - 32) )
    {
      v10 = *(_QWORD *)(v5 - 24);
      **(_QWORD **)(v5 - 16) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v5 - 16);
    }
    *(_QWORD *)(v5 - 32) = v1;
    if ( v1 )
    {
      v11 = *(_QWORD *)(v1 + 16);
      *(_QWORD *)(v5 - 24) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = v5 - 24;
      *(_QWORD *)(v5 - 16) = v1 + 16;
      *(_QWORD *)(v1 + 16) = v5 - 32;
    }
    sub_BD6B50(v5, v13);
  }
  return v5;
}
