// Function: sub_2901500
// Address: 0x2901500
//
__int64 __fastcall sub_2901500(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // eax
  __int64 v6; // rsi
  int v7; // ecx
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 v10; // rax
  __int64 v11; // r9
  unsigned int v12; // r12d
  int v13; // r11d
  __int64 v14; // r10
  _QWORD v15[4]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v16[5]; // [rsp+20h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a2 + 16);
    v7 = v4 - 1;
    v8 = *(_QWORD *)(a1 + 8);
    v15[2] = -4096;
    v16[2] = -8192;
    v15[0] = 0;
    v15[1] = 0;
    v16[0] = 0;
    v9 = (v4 - 1) & (((unsigned int)v6 >> 4) ^ ((unsigned int)v6 >> 9));
    v16[1] = 0;
    v10 = v8 + 32LL * v9;
    v11 = *(_QWORD *)(v10 + 16);
    if ( v6 == v11 )
    {
LABEL_4:
      *a3 = v10;
      v12 = 1;
    }
    else
    {
      v13 = 1;
      v14 = 0;
      while ( v11 != -4096 )
      {
        if ( v11 == -8192 && !v14 )
          v14 = v10;
        v9 = v7 & (v13 + v9);
        v10 = v8 + 32LL * v9;
        v11 = *(_QWORD *)(v10 + 16);
        if ( v11 == v6 )
          goto LABEL_4;
        ++v13;
      }
      if ( !v14 )
        v14 = v10;
      v12 = 0;
      *a3 = v14;
    }
    sub_D68D70(v16);
    sub_D68D70(v15);
    return v12;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
