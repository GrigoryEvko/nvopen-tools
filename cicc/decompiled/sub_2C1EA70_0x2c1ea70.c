// Function: sub_2C1EA70
// Address: 0x2c1ea70
//
__int64 __fastcall sub_2C1EA70(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v16; // rdx
  int v17; // eax
  int v18; // [rsp+Ch] [rbp-C4h]
  int v19; // [rsp+10h] [rbp-C0h]
  __int16 v20; // [rsp+14h] [rbp-BCh]
  unsigned __int8 v21; // [rsp+17h] [rbp-B9h]
  __int64 v22; // [rsp+18h] [rbp-B8h]
  __int64 v23; // [rsp+20h] [rbp-B0h]
  __int64 v24; // [rsp+20h] [rbp-B0h]
  __int64 v25; // [rsp+28h] [rbp-A8h]
  unsigned int v26; // [rsp+38h] [rbp-98h]
  __int64 v27; // [rsp+40h] [rbp-90h] BYREF
  int v28; // [rsp+48h] [rbp-88h]
  __int64 v29; // [rsp+50h] [rbp-80h]
  __int64 v30; // [rsp+58h] [rbp-78h]
  int v31; // [rsp+60h] [rbp-70h]
  char v32; // [rsp+64h] [rbp-6Ch]
  __int64 v33[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v34; // [rsp+90h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 904);
  v4 = *(_QWORD *)(a1 + 152);
  v19 = *(_DWORD *)(v3 + 104);
  v22 = *(_QWORD *)(v3 + 96);
  v21 = *(_BYTE *)(v3 + 110);
  v20 = *(_WORD *)(v3 + 108);
  *(_DWORD *)(v3 + 104) = *(_DWORD *)(v4 + 44);
  v18 = *(_DWORD *)(v4 + 40);
  v25 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 1);
  v5 = sub_2BFB640(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL), 0);
  BYTE4(v33[0]) = 0;
  v23 = v5;
  v6 = *(_QWORD *)(a1 + 48);
  LODWORD(v33[0]) = 0;
  v7 = sub_2BFB120(a2, *(_QWORD *)(v6 + 16), (unsigned int *)v33);
  v8 = *(_BYTE *)(a1 + 161) == 0;
  v32 = 0;
  v27 = v3;
  v28 = 0;
  v29 = 0;
  v31 = 0;
  v30 = v7;
  if ( v8 || (v9 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1))) == 0 )
  {
    v34 = 257;
    v13 = sub_ACD6D0(*(__int64 **)(v3 + 72));
    v10 = sub_B37620((unsigned int **)v3, *(_QWORD *)(a2 + 8), v13, v33);
  }
  else
  {
    v10 = sub_2BFB640(a2, v9, 0);
  }
  v8 = *(_BYTE *)(a1 + 160) == 0;
  v29 = v10;
  if ( v8 )
  {
    v16 = sub_F70320((__int64)&v27, v23, v4, v11, v12);
    if ( (unsigned int)(v18 - 6) <= 3 || (unsigned int)(v18 - 12) <= 3 )
    {
      v14 = sub_F6F180(v3, v18, v16, v25);
    }
    else
    {
      v24 = v16;
      v34 = 257;
      v17 = sub_1022EF0(*(_DWORD *)(v4 + 40));
      v14 = sub_2C137C0(v3, v17, v24, v25, v26, (__int64)v33, 0);
    }
  }
  else
  {
    v14 = sub_F70440((__int64)&v27, v4, v23, v25);
  }
  sub_2BF26E0(a2, a1 + 96, v14, 1);
  *(_QWORD *)(v3 + 96) = v22;
  *(_DWORD *)(v3 + 104) = v19;
  *(_WORD *)(v3 + 108) = v20;
  *(_BYTE *)(v3 + 110) = v21;
  return v21;
}
