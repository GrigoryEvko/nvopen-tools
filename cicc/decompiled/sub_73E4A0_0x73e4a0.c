// Function: sub_73E4A0
// Address: 0x73e4a0
//
__int64 __fastcall sub_73E4A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdi
  char v4; // dl
  __int64 v5; // rax
  int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // rbx
  _QWORD *v11; // r14
  __m128i *v12; // rax
  __int64 v13; // rax
  _BYTE *v14; // rax

  v2 = a1;
  if ( *(_BYTE *)(a1 + 24) )
  {
    v3 = *(_QWORD *)a1;
    v4 = *(_BYTE *)(v3 + 140);
    if ( v4 == 12 )
    {
      v5 = v3;
      do
      {
        v5 = *(_QWORD *)(v5 + 160);
        v4 = *(_BYTE *)(v5 + 140);
      }
      while ( v4 == 12 );
    }
    if ( v4 )
    {
      v7 = 0;
      v8 = sub_8D46C0(v3);
      if ( (*(_BYTE *)(v8 + 140) & 0xFB) == 8 )
        v7 = sub_8D4C10(v8, dword_4F077C4 != 2);
      v9 = *(_QWORD *)(a2 + 112);
      v10 = *(_QWORD **)(v9 + 16);
      v11 = v10;
      if ( (*(_BYTE *)(a2 + 96) & 2) == 0 )
        v11 = *(_QWORD **)(v9 + 8);
      for ( ; (_QWORD *)*v10 != v11; v11 = (_QWORD *)*v11 )
      {
        v12 = sub_73C570(*(const __m128i **)(v11[2] + 40LL), v7);
        v13 = sub_72D2E0(v12);
        v14 = sub_73DBF0(0xEu, v13, v2);
        v14[27] |= 2u;
        v2 = (__int64)v14;
      }
    }
  }
  return v2;
}
