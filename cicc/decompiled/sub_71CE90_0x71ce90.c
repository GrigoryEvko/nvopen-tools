// Function: sub_71CE90
// Address: 0x71ce90
//
__int64 __fastcall sub_71CE90(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v3; // r13
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 j; // rax
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 result; // rax
  __int64 i; // rax
  __int64 v19; // r14
  __int64 v20; // r12
  __int64 v21; // [rsp+8h] [rbp-48h]
  _QWORD v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = 0xFFFFFFFFLL;
  *(_BYTE *)(a1 + 172) = 0;
  v3 = sub_8600D0(17, 0xFFFFFFFFLL, 0, a1);
  *(_QWORD *)(v3 + 64) = sub_71B620(*(_QWORD *)(a1 + 152));
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 152) + 168LL) + 8LL) = a1;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  v5 = sub_72F130(v4);
  v6 = *(_QWORD *)(a1 + 248);
  if ( !v6 || *(_BYTE *)(v6 + 120) == 4 )
  {
    for ( i = *(_QWORD *)(v5 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
      v5 = *(_QWORD *)(a1 + 112);
  }
  else if ( v5 )
  {
    v7 = **(_QWORD **)(v5 + 248);
    v22[0] = sub_72F240(*(_QWORD *)(a1 + 240));
    v8 = sub_8B74F0(v7, v22, 0, dword_4F07508);
    v1 = 1;
    v9 = *(_QWORD *)(v8 + 88);
    v21 = *(_QWORD *)(*(_QWORD *)(v9 + 152) + 160LL);
    sub_8AD0D0(v8, 1, 1);
    for ( j = *(_QWORD *)(v9 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v5 = v9;
    if ( *(_QWORD *)(*(_QWORD *)(j + 168) + 40LL) )
    {
      v11 = **(_QWORD **)(*(_QWORD *)(a1 + 248) + 112LL);
      v1 = (__int64)v22;
      v22[0] = sub_72F240(*(_QWORD *)(a1 + 240));
      v5 = *(_QWORD *)(sub_8B74F0(v11, v22, 0, dword_4F07508) + 88);
      *(_QWORD *)(*(_QWORD *)(v5 + 152) + 160LL) = v21;
      *(_QWORD *)(v5 + 176) = v9;
      *(_BYTE *)(v5 + 172) = *(_BYTE *)(v9 + 172);
    }
  }
  else
  {
    v19 = sub_72C930(v4);
    v20 = **(_QWORD **)(*(_QWORD *)(a1 + 248) + 112LL);
    v1 = (__int64)v22;
    v22[0] = sub_72F240(*(_QWORD *)(a1 + 240));
    v5 = *(_QWORD *)(sub_8B74F0(v20, v22, 0, dword_4F07508) + 88);
    *(_QWORD *)(*(_QWORD *)(v5 + 152) + 160LL) = v19;
    *(_QWORD *)(v5 + 176) = 0;
  }
  v12 = sub_726B30(8);
  *(_QWORD *)(v12 + 48) = sub_731330(v5);
  sub_732AE0(v5);
  v13 = sub_726B30(11);
  v14 = *(_QWORD *)(v13 + 80);
  *(_QWORD *)(v13 + 72) = v12;
  *(_BYTE *)(v14 + 24) &= ~1u;
  *(_QWORD *)(v3 + 80) = v13;
  sub_863FC0(11, v1, v14, v15, v16);
  if ( dword_4F068EC && (*(_BYTE *)(a1 + 193) & 4) == 0 )
    sub_89A080(a1);
  result = (unsigned int)dword_4D041E0;
  if ( dword_4D041E0 )
  {
    *(_BYTE *)(a1 + 193) |= 2u;
    *(_BYTE *)(v3 + 29) |= 8u;
    *(_BYTE *)(v5 + 193) |= 2u;
  }
  return result;
}
