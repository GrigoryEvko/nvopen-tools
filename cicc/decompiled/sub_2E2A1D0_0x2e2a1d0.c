// Function: sub_2E2A1D0
// Address: 0x2e2a1d0
//
void __fastcall sub_2E2A1D0(__m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  _BYTE *v8; // rsi
  __int64 v9; // [rsp+8h] [rbp-18h] BYREF

  v7 = (_QWORD *)sub_2E29D60(a1, a2, a3, a4, a5, a6);
  if ( (_QWORD *)*v7 == v7 )
  {
    v9 = a3;
    v8 = (_BYTE *)v7[5];
    if ( v8 == (_BYTE *)v7[6] )
    {
      sub_2E26050((__int64)(v7 + 4), v8, &v9);
    }
    else
    {
      if ( v8 )
      {
        *(_QWORD *)v8 = a3;
        v8 = (_BYTE *)v7[5];
      }
      v7[5] = v8 + 8;
    }
  }
}
