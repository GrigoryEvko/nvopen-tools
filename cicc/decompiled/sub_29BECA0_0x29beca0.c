// Function: sub_29BECA0
// Address: 0x29beca0
//
void __fastcall sub_29BECA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // r13
  _QWORD *v8; // r12
  __int16 v9; // dx
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h]
  unsigned __int64 v16; // [rsp+28h] [rbp-38h]

  v14 = a1 + 48;
  v5 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (_QWORD *)(a1 + 48) != v5 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v5 - 3;
        v16 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
        v5 = (_QWORD *)v16;
        v10 = sub_AA5030(a2, 1);
        if ( v10 )
        {
          v15 = v10;
          v10 -= 24;
          v11 = (unsigned __int8)v9;
          BYTE1(v11) = HIBYTE(v9);
        }
        else
        {
          v15 = 0;
          v11 = 0;
        }
        if ( (unsigned __int8)sub_29BDD80(v8, v10, a3, a4, a5, 0) )
          break;
        if ( v14 == v16 )
          return;
      }
      sub_B44500(v8, v15, v11);
    }
    while ( v14 != v16 );
  }
}
