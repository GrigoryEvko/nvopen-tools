// Function: sub_15E93A0
// Address: 0x15e93a0
//
__int64 __fastcall sub_15E93A0(__int64 a1, __int64 a2, __int64 a3, __m128i a4)
{
  __int64 v5; // rbx
  __int64 i; // r14
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx

  sub_16E7EE0(*(_QWORD *)a2, *(const char **)(a2 + 8), *(_QWORD *)(a2 + 16));
  if ( (unsigned __int8)sub_160E740("*", 1) )
  {
    sub_155BB10(a3, *(_QWORD *)a2, 0, *(_BYTE *)(a2 + 40), 0, a4);
  }
  else
  {
    v5 = *(_QWORD *)(a3 + 32);
    for ( i = a3 + 24; i != v5; v5 = *(_QWORD *)(v5 + 8) )
    {
      while ( 1 )
      {
        v7 = 0;
        if ( v5 )
          v7 = v5 - 56;
        v8 = sub_1649960(v7);
        if ( (unsigned __int8)sub_160E740(v8, v9) )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( i == v5 )
          goto LABEL_9;
      }
      sub_1559E80(v7, *(_QWORD *)a2, 0, 0, 0);
    }
  }
LABEL_9:
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 40;
  *(_QWORD *)(a1 + 16) = a1 + 40;
  *(_QWORD *)(a1 + 64) = a1 + 96;
  *(_QWORD *)(a1 + 72) = a1 + 96;
  *(_QWORD *)(a1 + 24) = 0x100000002LL;
  *(_QWORD *)(a1 + 80) = 2;
  *(_QWORD *)(a1 + 40) = &unk_4F9EE48;
  *(_DWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = 1;
  return a1;
}
