// Function: sub_38D4500
// Address: 0x38d4500
//
__int64 __fastcall sub_38D4500(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 *v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 result; // rax
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 72LL);
  if ( (char *)v2 == (char *)sub_38D3BD0 )
  {
    LODWORD(v3) = 0;
    if ( *(_BYTE *)(a1 + 260) )
      v3 = *(_QWORD *)(a1 + 264);
  }
  else
  {
    LODWORD(v3) = v2();
  }
  if ( sub_38CF2B0(a2, v13, v3) )
    return sub_38DCDD0(a1, v13[0]);
  v4 = sub_22077B0(0x58u);
  v5 = v4;
  if ( v4 )
  {
    v6 = v4;
    sub_38CF760(v4, 8, 0, 0);
    *(_QWORD *)(v5 + 48) = a2;
    *(_QWORD *)(v5 + 64) = v5 + 80;
    *(_BYTE *)(v5 + 56) = 0;
    *(_QWORD *)(v5 + 72) = 0x800000001LL;
    *(_BYTE *)(v5 + 80) = 0;
  }
  else
  {
    v6 = 0;
  }
  sub_38D4150(a1, v5, 0);
  v7 = *(unsigned int *)(a1 + 120);
  v8 = 0;
  if ( (_DWORD)v7 )
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v7 - 32);
  v9 = *(__int64 **)(a1 + 272);
  v10 = *v9;
  v11 = *(_QWORD *)v5 & 7LL;
  *(_QWORD *)(v5 + 8) = v9;
  v10 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v5 = v10 | v11;
  *(_QWORD *)(v10 + 8) = v6;
  result = *v9 & 7;
  *v9 = result | v6;
  *(_QWORD *)(v5 + 24) = v8;
  return result;
}
