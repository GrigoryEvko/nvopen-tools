// Function: sub_148F000
// Address: 0x148f000
//
__int64 __fastcall sub_148F000(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  int v11; // edx
  int v12; // r10d
  __int64 v13; // [rsp+8h] [rbp-58h] BYREF
  __int64 v14; // [rsp+18h] [rbp-48h] BYREF
  _BYTE v15[16]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v16; // [rsp+30h] [rbp-30h]

  v5 = *(unsigned int *)(a1 + 32);
  v13 = a2;
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a1 + 16);
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
        return v8[1];
    }
    else
    {
      v11 = 1;
      while ( v9 != -8 )
      {
        v12 = v11 + 1;
        v7 = (v5 - 1) & (v11 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v11 = v12;
      }
    }
  }
  v14 = sub_148EA50((__int64 *)a1, a2, a3, a4);
  sub_1466830((__int64)v15, a1 + 8, &v13, &v14);
  return *(_QWORD *)(v16 + 8);
}
