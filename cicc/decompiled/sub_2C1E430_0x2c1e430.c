// Function: sub_2C1E430
// Address: 0x2c1e430
//
__int64 *__fastcall sub_2C1E430(__int64 a1, __int64 a2)
{
  unsigned int **v3; // r14
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD *v6; // r9
  int v7; // edx
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // r15
  unsigned int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rax
  int v15; // [rsp+0h] [rbp-80h]
  __int64 v16; // [rsp+8h] [rbp-78h]
  _BYTE *v17; // [rsp+18h] [rbp-68h] BYREF
  __int64 v18[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]

  v3 = *(unsigned int ***)(a2 + 904);
  v18[0] = *(_QWORD *)(a1 + 88);
  if ( v18[0] )
    sub_2AAAFA0(v18);
  sub_2BF1A90(a2, (__int64)v18);
  sub_9C6650(v18);
  if ( *(_DWORD *)(a1 + 56) == 2 && (v4 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL)) != 0 )
  {
    v5 = *(_QWORD *)(v4 + 40);
    v6 = *(_QWORD **)(v5 + 24);
    if ( *(_DWORD *)(v5 + 32) > 0x40u )
      v6 = (_QWORD *)*v6;
    v7 = (int)v6;
  }
  else
  {
    LODWORD(v6) = 0;
    v7 = 0;
  }
  v15 = (int)v6;
  v8 = sub_2C0D490(*(_BYTE *)(a2 + 12), 0, v7, (__int64)v3);
  BYTE4(v18[0]) = 0;
  v16 = v8;
  v9 = *(__int64 **)(a1 + 48);
  LODWORD(v18[0]) = 0;
  v10 = sub_2BFB120(a2, *v9, (unsigned int *)v18);
  v17 = (_BYTE *)sub_2AB26E0((__int64)v3, v16, *(_QWORD *)(a2 + 8), v15);
  v11 = *(_DWORD *)(a1 + 156);
  v12 = *(_QWORD *)(a1 + 160);
  v19 = 257;
  v13 = sub_921130(v3, v12, v10, &v17, 1, (__int64)v18, v11);
  return sub_2BF26E0(a2, a1 + 96, v13, 1);
}
