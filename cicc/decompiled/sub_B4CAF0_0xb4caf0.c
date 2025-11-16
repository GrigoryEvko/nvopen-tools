// Function: sub_B4CAF0
// Address: 0xb4caf0
//
__int64 __fastcall sub_B4CAF0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx

  v4 = sub_BD5C60(a2, a2);
  v5 = sub_BCB120(v4);
  sub_B44260(a1, v5, 2, a3, 0, 0);
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 1 )
  {
    v10 = *(_QWORD *)(a2 - 96);
    if ( *(_QWORD *)(a1 - 96) )
    {
      v11 = *(_QWORD *)(a1 - 88);
      **(_QWORD **)(a1 - 80) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = *(_QWORD *)(a1 - 80);
    }
    *(_QWORD *)(a1 - 96) = v10;
    if ( v10 )
    {
      v12 = *(_QWORD *)(v10 + 16);
      *(_QWORD *)(a1 - 88) = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = a1 - 88;
      *(_QWORD *)(a1 - 80) = v10 + 16;
      *(_QWORD *)(v10 + 16) = a1 - 96;
    }
    v13 = *(_QWORD *)(a2 - 64);
    if ( *(_QWORD *)(a1 - 64) )
    {
      v14 = *(_QWORD *)(a1 - 56);
      **(_QWORD **)(a1 - 48) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = *(_QWORD *)(a1 - 48);
    }
    *(_QWORD *)(a1 - 64) = v13;
    if ( v13 )
    {
      v15 = *(_QWORD *)(v13 + 16);
      *(_QWORD *)(a1 - 56) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = a1 - 56;
      *(_QWORD *)(a1 - 48) = v13 + 16;
      *(_QWORD *)(v13 + 16) = a1 - 64;
    }
  }
  v6 = *(_QWORD *)(a2 - 32);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v7 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = v6;
  if ( v6 )
  {
    v8 = *(_QWORD *)(v6 + 16);
    *(_QWORD *)(a1 - 24) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = v6 + 16;
    *(_QWORD *)(v6 + 16) = a1 - 32;
  }
  result = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1u;
  *(_BYTE *)(a1 + 1) = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1;
  return result;
}
