// Function: sub_2D612E0
// Address: 0x2d612e0
//
__int64 __fastcall sub_2D612E0(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rdi
  __int64 v5; // rax
  unsigned int v6; // r13d
  __int64 *v8; // rax
  unsigned __int8 *v9; // r14
  unsigned int v10; // eax
  unsigned __int8 *v11; // rdx
  char v12; // si
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // [rsp+0h] [rbp-120h]
  unsigned __int8 *v20; // [rsp+18h] [rbp-108h]
  int v21; // [rsp+28h] [rbp-F8h]
  _BYTE v22[32]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v23; // [rsp+50h] [rbp-D0h]
  unsigned int *v24[24]; // [rsp+60h] [rbp-C0h] BYREF

  if ( (unsigned int)*(unsigned __int8 *)(a2[1] + 8LL) - 17 > 1 )
    return 0;
  if ( !(unsigned __int8)sub_DFE6D0(*(_QWORD *)(a1 + 32)) )
    return 0;
  v4 = *(a2 - 4);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 )
    return 0;
  if ( !*(_QWORD *)(v5 + 8)
    && *(_BYTE *)v4 == 86
    && (v8 = (__int64 *)sub_986520(v4), (v19 = *v8) != 0)
    && (v9 = (unsigned __int8 *)v8[4]) != 0
    && (v20 = (unsigned __int8 *)v8[8]) != 0
    && sub_9B7DA0((char *)v9, 0xFFFFFFFF, 0)
    && (LOBYTE(v10) = sub_9B7DA0((char *)v20, 0xFFFFFFFF, 0), v6 = v10, (_BYTE)v10) )
  {
    sub_23D0AB0((__int64)v24, (__int64)a2, 0, 0, 0);
    v11 = (unsigned __int8 *)*(a2 - 8);
    v12 = *(_BYTE *)a2 - 29;
    v23 = 257;
    v13 = sub_2D5B950((__int64 *)v24, v12, v11, v9, v21, 0, (__int64)v22, 0);
    v23 = 257;
    v14 = v13;
    v15 = sub_2D5B950((__int64 *)v24, v12, (unsigned __int8 *)*(a2 - 8), v20, v21, 0, (__int64)v22, 0);
    v23 = 257;
    v16 = sub_B36550(v24, v19, v14, v15, (__int64)v22, 0);
    sub_2D594F0((__int64)a2, v16, (__int64 *)(a1 + 840), *(unsigned __int8 *)(a1 + 832), v17, v18);
    sub_B43D60(a2);
    sub_F94A20(v24, v16);
  }
  else
  {
    return 0;
  }
  return v6;
}
