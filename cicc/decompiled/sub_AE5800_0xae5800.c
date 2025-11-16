// Function: sub_AE5800
// Address: 0xae5800
//
__int64 __fastcall sub_AE5800(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 **a4)
{
  __int64 v6; // r14
  unsigned __int8 v7; // al
  __int64 v8; // r14
  char v9; // bl
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // eax
  __int64 v14; // r15
  unsigned __int64 v15; // rsi
  char v16; // al
  char v17; // al
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  int v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  __int64 v23; // [rsp+28h] [rbp-38h]

  v6 = *a3;
  v7 = *(_BYTE *)(*a3 + 8);
  if ( v7 == 16 )
  {
    v8 = *(_QWORD *)(v6 + 24);
    *a3 = v8;
    v9 = sub_AE5020(a2, v8);
    v10 = sub_9208B0(a2, v8);
    v23 = v11;
    v22 = ((1LL << v9) + ((unsigned __int64)(v10 + 7) >> 3) - 1) >> v9 << v9;
    sub_AE1360((__int64)&v20, v22, v11, (__int64 *)a4);
    v12 = v21;
    *(_BYTE *)(a1 + 16) = 1;
    *(_DWORD *)(a1 + 8) = v12;
    *(_QWORD *)a1 = v20;
  }
  else if ( (unsigned int)v7 - 17 > 1
         && v7 == 15
         && ((v14 = sub_AE4AC0(a2, v6), *((_DWORD *)a4 + 2) <= 0x40u) ? (v15 = (unsigned __int64)*a4) : (v15 = **a4),
             v16 = *(_BYTE *)(v14 + 8),
             v22 = *(_QWORD *)v14,
             LOBYTE(v23) = v16,
             sub_CA1930(&v22) > v15) )
  {
    v19 = (unsigned int)sub_AE1C80(v14, v15);
    v17 = *(_BYTE *)(v14 + 16 * v19 + 32);
    v22 = *(_QWORD *)(v14 + 16 * v19 + 24);
    LOBYTE(v23) = v17;
    v18 = sub_CA1930(&v22);
    sub_C46F20(a4, v18);
    *a3 = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 8 * v19);
    *(_DWORD *)(a1 + 8) = 32;
    *(_QWORD *)a1 = v19;
    *(_BYTE *)(a1 + 16) = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 16) = 0;
  }
  return a1;
}
