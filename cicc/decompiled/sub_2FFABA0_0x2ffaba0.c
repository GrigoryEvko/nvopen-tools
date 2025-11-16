// Function: sub_2FFABA0
// Address: 0x2ffaba0
//
void __fastcall sub_2FFABA0(__int64 a1, unsigned int a2, __int64 a3, char a4)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v9; // rax
  _BYTE *v10; // rsi
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_2E8F690(a3, a2, *(_QWORD **)(a1 + 96), a4) )
  {
    v9 = sub_2E29D60((__m128i *)a1, a2, v5, v6, v7, v8);
    v11[0] = a3;
    v10 = *(_BYTE **)(v9 + 40);
    if ( v10 == *(_BYTE **)(v9 + 48) )
    {
      sub_2E26050(v9 + 32, v10, v11);
    }
    else
    {
      if ( v10 )
      {
        *(_QWORD *)v10 = a3;
        v10 = *(_BYTE **)(v9 + 40);
      }
      *(_QWORD *)(v9 + 40) = v10 + 8;
    }
  }
}
