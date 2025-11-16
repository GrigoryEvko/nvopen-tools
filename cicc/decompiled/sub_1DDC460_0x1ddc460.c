// Function: sub_1DDC460
// Address: 0x1ddc460
//
__int64 __fastcall sub_1DDC460(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  int v5; // ecx
  unsigned int v6; // eax
  int v7; // edi
  __int64 v8; // r9
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r10
  __int64 v12; // rdx
  int v14; // eax
  int v15; // r11d
  unsigned int v16[3]; // [rsp+Ch] [rbp-14h] BYREF

  v4 = *(_QWORD *)(a2 + 232);
  v5 = *(_DWORD *)(v4 + 184);
  v6 = -1;
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = *(_QWORD *)(v4 + 168);
    v9 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( a3 == *v10 )
    {
LABEL_3:
      v6 = *((_DWORD *)v10 + 2);
    }
    else
    {
      v14 = 1;
      while ( v11 != -8 )
      {
        v15 = v14 + 1;
        v9 = v7 & (v14 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( a3 == *v10 )
          goto LABEL_3;
        v14 = v15;
      }
      v6 = -1;
    }
  }
  v12 = **(_QWORD **)(v4 + 128);
  v16[0] = v6;
  sub_1370E50(a1, v4, v12, v16);
  return a1;
}
