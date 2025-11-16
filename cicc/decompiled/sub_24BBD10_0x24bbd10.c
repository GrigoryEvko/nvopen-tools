// Function: sub_24BBD10
// Address: 0x24bbd10
//
_QWORD *__fastcall sub_24BBD10(_QWORD *a1, __int64 a2, size_t a3, __int64 a4)
{
  char v6; // al
  _QWORD *v7; // rsi
  _QWORD *v8; // rdx
  __int64 v10; // [rsp+8h] [rbp-58h]
  __int64 v11; // [rsp+10h] [rbp-50h] BYREF
  __int64 v12; // [rsp+18h] [rbp-48h] BYREF
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  __int64 v14[7]; // [rsp+28h] [rbp-38h] BYREF

  v11 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v12 = v11;
  v13 = v11;
  v14[0] = v11;
  v10 = sub_BC0510(a4, &unk_4F87C68, a3);
  v6 = sub_24B8750(
         a3,
         *(_QWORD *)a2,
         *(_QWORD *)(a2 + 8),
         *(__m128i **)(a2 + 32),
         *(__int64 ***)(a2 + 40),
         *(__int64 **)(a2 + 72),
         (int)sub_24A2BC0,
         &v11,
         (int)sub_24A2BE0,
         &v12,
         (int)sub_24A2C00,
         &v13,
         (int)sub_24A2C20,
         v14,
         (__int64 *)(v10 + 8),
         *(_BYTE *)(a2 + 64));
  v7 = a1 + 4;
  v8 = a1 + 10;
  if ( v6 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v7;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v8;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v7;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v8;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  return a1;
}
