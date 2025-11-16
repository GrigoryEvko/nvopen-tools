// Function: sub_7D9F20
// Address: 0x7d9f20
//
void __fastcall sub_7D9F20(__int64 a1)
{
  __int64 v1; // r12
  const __m128i *v3; // rsi
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdi
  __int64 v10; // [rsp+8h] [rbp-88h] BYREF
  _BYTE v11[128]; // [rsp+10h] [rbp-80h] BYREF

  v1 = *(_QWORD *)(a1 + 16);
  if ( v1 )
  {
    if ( *(_BYTE *)(a1 + 32) )
    {
      v4 = qword_4F04C50;
      qword_4F04C50 = 0;
      v3 = (const __m128i *)unk_4F07288;
    }
    else
    {
      v3 = 0;
      v4 = 0;
    }
    sub_7E1A40(v11, v3, 0, &v10);
    if ( dword_4F077C4 == 2 )
      sub_7F2600(v1, 0);
    else
      sub_7D9EC0((_QWORD *)v1, v3, v5, v6, v7, v8);
    sub_7E1B40(v10);
    if ( *(_BYTE *)(a1 + 32) )
      qword_4F04C50 = v4;
    if ( *(_BYTE *)(v1 + 24) == 2 )
    {
      v9 = *(_QWORD *)(v1 + 56);
      if ( *(_BYTE *)(v9 + 173) == 1 && (int)sub_6210B0(v9, 0) <= 0 )
        sub_6851C0(0x5Eu, (_DWORD *)(a1 + 36));
    }
  }
}
