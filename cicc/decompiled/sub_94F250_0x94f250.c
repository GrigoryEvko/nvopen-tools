// Function: sub_94F250
// Address: 0x94f250
//
__int64 __fastcall sub_94F250(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // r14
  __m128i *v8; // rax
  __int64 v9; // rsi
  unsigned int **v10; // r12
  unsigned __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v15; // [rsp+0h] [rbp-A0h]
  __int64 v16; // [rsp+8h] [rbp-98h]
  int v17; // [rsp+1Ch] [rbp-84h] BYREF
  _QWORD v18[4]; // [rsp+20h] [rbp-80h] BYREF
  _BYTE v19[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v20; // [rsp+60h] [rbp-40h]

  v5 = *(_QWORD *)(*(_QWORD *)(a4 + 16) + 16LL);
  v15 = *(_QWORD *)(a4 + 16);
  v6 = *(_QWORD *)(v5 + 16);
  v16 = *(_QWORD *)(v6 + 16);
  v7 = sub_90A810(*(__int64 **)(a2 + 32), a3, 0, 0);
  v18[0] = sub_92F410(a2, v15);
  v8 = sub_92F410(a2, v5);
  v9 = v6;
  v10 = (unsigned int **)(a2 + 48);
  v18[1] = v8;
  v18[2] = sub_92F410(a2, v9);
  v11 = 0;
  v18[3] = sub_92F410(a2, v16);
  v20 = 257;
  if ( v7 )
    v11 = *(_QWORD *)(v7 + 24);
  v12 = sub_921880(v10, v11, v7, (int)v18, 4, (__int64)v19, 0);
  v17 = 0;
  v20 = 257;
  v13 = sub_94D3D0(v10, v12, (__int64)&v17, 1, (__int64)v19);
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v13;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
