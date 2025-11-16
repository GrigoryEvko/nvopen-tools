// Function: sub_315C560
// Address: 0x315c560
//
void __fastcall sub_315C560(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rsi

  v3 = a2 + 72;
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  sub_315A7B0((__int64 *)a1);
  v4 = *(_QWORD *)(a2 + 80);
  if ( v4 != a2 + 72 )
  {
    do
    {
      v5 = v4 - 24;
      if ( !v4 )
        v5 = 0;
      sub_3158140(a1, v5);
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v3 != v4 );
  }
}
