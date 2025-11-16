// Function: sub_38D4EE0
// Address: 0x38d4ee0
//
__int64 __fastcall sub_38D4EE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 result; // rax

  v5 = sub_38D4BB0(a1, 0);
  sub_38D4150(a1, v5, *(unsigned int *)(v5 + 72));
  v6 = sub_22077B0(0x50u);
  v7 = v6;
  if ( v6 )
  {
    v8 = v6;
    sub_38CF760(v6, 3, 0, 0);
    *(_QWORD *)(v7 + 48) = a3;
    *(_BYTE *)(v7 + 56) = 1;
    *(_QWORD *)(v7 + 64) = a2;
    *(_QWORD *)(v7 + 72) = a4;
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
  result = *v11 & 7;
  *v11 = result | v8;
  *(_QWORD *)(v7 + 24) = v10;
  return result;
}
