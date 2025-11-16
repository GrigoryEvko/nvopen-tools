// Function: sub_948130
// Address: 0x948130
//
__int64 __fastcall sub_948130(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v5; // r14
  __int64 v6; // r15
  __m128i *v7; // rbx
  int v8; // r9d
  int v9; // ecx
  unsigned __int64 v10; // rax
  __m128i *v11; // r15
  int v12; // eax
  bool v14; // al
  int v15; // [rsp+0h] [rbp-50h]
  int v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL);
  v5 = *(_QWORD *)(v4 + 16);
  v6 = *(_QWORD *)(v5 + 16);
  if ( a3 == 659 )
  {
    v19[0] = 4;
  }
  else if ( a3 <= 0x293 )
  {
    if ( a3 != 409 )
    {
      if ( a3 == 658 )
      {
        v19[0] = 2;
        goto LABEL_6;
      }
      goto LABEL_12;
    }
    v18 = *(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL);
    v14 = sub_91CB00(*(_QWORD *)(v6 + 16), (__int64 *)v19);
    v4 = v18;
    if ( !v14 )
      sub_91B8A0("align value for memcpy was not constant", (_DWORD *)(a4 + 36), 1);
  }
  else
  {
    if ( a3 != 660 )
    {
      if ( a3 == 661 )
      {
        v19[0] = 16;
        goto LABEL_6;
      }
LABEL_12:
      v19[0] = 1;
      goto LABEL_6;
    }
    v19[0] = 8;
  }
LABEL_6:
  v7 = sub_92F410(a2, v4);
  v8 = 0;
  v9 = 0;
  if ( v19[0] )
  {
    _BitScanReverse64(&v10, v19[0]);
    LOBYTE(v8) = 63 - (v10 ^ 0x3F);
    LOBYTE(v9) = v8;
    LODWORD(v10) = v8;
    BYTE1(v9) = 1;
    BYTE1(v10) = 1;
    v8 = v10;
  }
  v15 = v8;
  v17 = v9;
  v11 = sub_92F410(a2, v6);
  v12 = (unsigned int)sub_92F410(a2, v5);
  sub_92CB30(a2, (int)v7, v12, (__int64)v11, v17, v15, 0);
  *(_QWORD *)a1 = v7;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
