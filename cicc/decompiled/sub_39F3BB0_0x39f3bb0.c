// Function: sub_39F3BB0
// Address: 0x39f3bb0
//
__int64 __fastcall sub_39F3BB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax

  if ( sub_390B160(*(_QWORD *)(a1 + 264), a2) )
  {
    v6 = sub_22077B0(0xE0u);
    v7 = v6;
    if ( v6 )
    {
      v8 = v6;
      sub_38CF760(v6, 1, 0, 0);
      *(_QWORD *)(v7 + 56) = 0;
      *(_WORD *)(v7 + 48) = 0;
      *(_QWORD *)(v7 + 64) = v7 + 80;
      *(_QWORD *)(v7 + 72) = 0x2000000000LL;
      *(_QWORD *)(v7 + 112) = v7 + 128;
      *(_QWORD *)(v7 + 120) = 0x400000000LL;
    }
    else
    {
      v8 = 0;
    }
    sub_38D4150(a1, v7, 0);
    v9 = *(unsigned int *)(a1 + 120);
    v10 = 0;
    if ( (_DWORD)v9 )
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v9 - 32);
    v11 = *(__int64 **)(a1 + 272);
    v12 = *v11;
    v13 = *(_QWORD *)v7 & 7LL;
    *(_QWORD *)(v7 + 8) = v11;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v7 = v12 | v13;
    *(_QWORD *)(v12 + 8) = v8;
    *v11 = *v11 & 7 | v8;
    *(_QWORD *)(v7 + 24) = v10;
  }
  result = sub_38D6580(a1, a2, a3);
  *(_WORD *)(a2 + 12) &= 0xFFF8u;
  return result;
}
