// Function: sub_2495170
// Address: 0x2495170
//
__int64 __fastcall sub_2495170(__int64 a1, unsigned __int64 a2, _BYTE *a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  unsigned __int64 v10; // r10
  __int64 v11; // r11
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // r15
  __int64 v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r15
  __int64 **v21; // rax
  __int64 v22; // rdx
  const char *v24; // rax
  __int64 v25; // rdx
  char v26; // al
  unsigned int v29; // [rsp+10h] [rbp-A0h]
  _BYTE v30[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v31; // [rsp+40h] [rbp-70h]
  _BYTE v32[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v33; // [rsp+70h] [rbp-40h]

  if ( *(_BYTE *)a2 <= 0x15u )
    return (__int64)a3;
  v10 = a5;
  v11 = a6;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    v12 = sub_B43CB0(a2);
    v10 = a5;
    v11 = a6;
    if ( *(_BYTE *)(a1 + 488) )
    {
      v24 = sub_BD5D20(v12);
      v26 = sub_C89090((_QWORD *)(a1 + 472), v24, v25, 0, 0);
      v10 = a5;
      v11 = a6;
      if ( !v26 )
        return (__int64)a3;
    }
  }
  v13 = sub_24941A0(a1, a2, a3, a4, v10, v11);
  v14 = *(_QWORD **)(a4 + 72);
  v15 = v13;
  v33 = 257;
  v16 = sub_BCB2D0(v14);
  v17 = (_BYTE *)sub_ACD640(v16, 1, 0);
  v18 = sub_92B530((unsigned int **)a4, 0x20u, v15, v17, (__int64)v32);
  v19 = *(_QWORD *)(a2 + 8);
  v33 = 257;
  v20 = v18;
  v31 = 257;
  v21 = (__int64 **)sub_2491640((_QWORD *)(a1 + 16), v19);
  if ( *(_BYTE *)(a4 + 108) )
    v22 = sub_B358C0(a4, 0x6Eu, a2, (__int64)v21, v29, (__int64)v30, 0, 0, 0);
  else
    v22 = sub_24932B0((__int64 *)a4, 0x2Eu, a2, v21, (__int64)v30, 0, v29, 0);
  return sub_B36550((unsigned int **)a4, v20, v22, (__int64)a3, (__int64)v32, 0);
}
