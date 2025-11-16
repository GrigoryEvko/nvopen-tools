// Function: sub_7E7D10
// Address: 0x7e7d10
//
__int64 __fastcall sub_7E7D10(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // r13
  __int64 result; // rax
  __int64 v4; // rbx
  _BYTE *v6; // rax
  __int64 v7; // rdi
  void *v8; // r13
  __int64 v9; // rsi
  __m128i *v10; // r13
  __m128i *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r14
  __int64 v19; // r13
  __int64 *v20; // rax

  v2 = (_BYTE *)a1[9];
  result = *((_QWORD *)v2 + 2);
  v4 = *(_QWORD *)(result + 56);
  if ( (*(_BYTE *)(v4 + 146) & 8) != 0 )
  {
    *((_QWORD *)v2 + 2) = 0;
    if ( *((_BYTE *)a1 + 56) == 94 )
    {
      if ( (v2[25] & 1) != 0 )
      {
        v2 = sub_73E1B0((__int64)v2, a2);
      }
      else
      {
        v17 = (__int64)v2;
        v18 = sub_7E7CB0(*(_QWORD *)v2);
        v19 = sub_7E2BE0(v18, (__int64)v2);
        v20 = (__int64 *)sub_73E230(v18, v17);
        v2 = sub_73DF90(v19, v20);
      }
    }
    v6 = sub_7E23D0(v2);
    v7 = *(_QWORD *)(v4 + 128);
    v8 = v6;
    if ( v7 )
    {
      *((_QWORD *)v6 + 2) = sub_73A830(v7, byte_4F06A51[0]);
      v16 = sub_7E1C30();
      v8 = sub_73DBF0(0x32u, v16, (__int64)v8);
    }
    v9 = sub_72D2E0((_QWORD *)*a1);
    v10 = (__m128i *)sub_73E110((__int64)v8, v9);
    if ( (*((_BYTE *)a1 + 25) & 1) != 0 || (unsigned int)sub_8D3A70(*a1) )
    {
      v11 = (__m128i *)sub_73DCD0(v10);
      v10 = v11;
      if ( (*((_BYTE *)a1 + 25) & 1) == 0 )
        v10 = (__m128i *)sub_731370((__int64)v11, v9, v12, v13, v14, v15);
    }
    return sub_730620((__int64)a1, v10);
  }
  return result;
}
