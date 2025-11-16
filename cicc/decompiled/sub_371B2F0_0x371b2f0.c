// Function: sub_371B2F0
// Address: 0x371b2f0
//
__int64 __fastcall sub_371B2F0(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rdx
  char v8; // si
  __int64 v9; // rax
  char v10; // cl
  __int64 v11; // rax
  __int16 v13; // ax

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL);
  v3 = *(_QWORD *)a1 + 48LL;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v2;
  if ( v2 != v3 )
  {
    v4 = *(_QWORD *)(a1 + 24);
    if ( v2 )
      v2 -= 24;
    v5 = sub_3186770(v4, v2);
    v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 88LL))(v5);
    v7 = *(_QWORD *)(a1 + 8);
    v8 = *(_BYTE *)(a1 + 16);
    v9 = (unsigned int)(v6 - 1);
    v10 = *(_BYTE *)(a1 + 17);
    if ( v9 )
    {
      v11 = v9 - 1;
      do
        v7 = *(_QWORD *)(v7 + 8);
      while ( v11-- != 0 );
      v10 = 0;
      v8 = 0;
    }
    *(_QWORD *)(a1 + 8) = v7;
    LOBYTE(v13) = v8;
    HIBYTE(v13) = v10;
    *(_WORD *)(a1 + 16) = v13;
  }
  return a1;
}
