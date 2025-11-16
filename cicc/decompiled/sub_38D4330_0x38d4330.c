// Function: sub_38D4330
// Address: 0x38d4330
//
__int64 __fastcall sub_38D4330(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 *v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rdx

  v6 = sub_22077B0(0x48u);
  v7 = v6;
  if ( v6 )
  {
    v8 = v6;
    sub_38CF760(v6, 5, 0, 0);
    *(_QWORD *)(v7 + 48) = a2;
    *(_BYTE *)(v7 + 56) = a3;
    *(_QWORD *)(v7 + 64) = a4;
  }
  else
  {
    v8 = 0;
  }
  result = sub_38D4150(a1, v7, 0);
  v10 = *(unsigned int *)(a1 + 120);
  v11 = 0;
  if ( (_DWORD)v10 )
    v11 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v10 - 32);
  v12 = *(__int64 **)(a1 + 272);
  v13 = *v12;
  v14 = *(_QWORD *)v7 & 7LL;
  *(_QWORD *)(v7 + 8) = v12;
  v13 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v7 = v13 | v14;
  *(_QWORD *)(v13 + 8) = v8;
  *v12 = *v12 & 7 | v8;
  *(_QWORD *)(v7 + 24) = v11;
  return result;
}
