// Function: sub_609BF0
// Address: 0x609bf0
//
__int64 __fastcall sub_609BF0(__int64 a1)
{
  __m128i *v2; // r9
  __int64 v3; // rbx
  __int64 v4; // rcx
  _QWORD *v5; // r8
  __int64 v6; // rdi
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // rdx
  bool v10; // zf
  __int64 v11; // r12
  int v13; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v14[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = (__m128i *)(a1 + 344);
  v3 = *(_QWORD *)a1;
  v4 = *(_QWORD *)(a1 + 336);
  v5 = **(_QWORD ***)(a1 + 192);
  v6 = *(_QWORD *)(qword_4F04C68[0]
                 + 776LL * *(int *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 240) + 168LL) + 152LL) + 240LL)
                 + 600);
  *(_BYTE *)(v3 + 121) |= 0x40u;
  v7 = 1;
  if ( !unk_4F0775C )
  {
    v7 = 0;
    if ( dword_4F077B4 )
    {
      if ( dword_4F077C4 == 2 )
        v7 = unk_4F077A0 != 0;
    }
  }
  *(_BYTE *)(v3 + 124) = (v7 << 6) | *(_BYTE *)(v3 + 124) & 0xBF;
  v8 = sub_6040F0(v6, a1, 1, 0, v5, &v13, v14, 0, v4, v2);
  v10 = qword_4CF8008 == 0;
  *(_QWORD *)(v3 + 232) = 0;
  v11 = v8;
  if ( !v10 )
    sub_5E9610(v6, a1, v9);
  if ( !v11 )
    return v11;
  if ( (*(_BYTE *)(v11 + 81) & 0x20) != 0 )
    return 0;
  if ( (unsigned __int8)(*(_BYTE *)(v11 + 80) - 20) <= 1u )
    return v11;
  sub_6854C0(786, v11 + 48, v11);
  return 0;
}
