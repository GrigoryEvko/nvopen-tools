// Function: sub_375BC20
// Address: 0x375bc20
//
void __fastcall sub_375BC20(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 v6; // r15
  __int64 v9; // rax
  unsigned __int16 v10; // dx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  char v14; // al
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v21; // [rsp+10h] [rbp-50h] BYREF
  __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h]
  __int64 v24; // [rsp+28h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  LOWORD(v21) = v10;
  v22 = v11;
  if ( v10 )
  {
    if ( v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
      BUG();
    v19 = 16LL * (v10 - 1);
    v13 = *(_QWORD *)&byte_444C4A0[v19];
    v14 = byte_444C4A0[v19 + 8];
  }
  else
  {
    v23 = sub_3007260((__int64)&v21);
    v24 = v12;
    v13 = v23;
    v14 = v24;
  }
  v21 = v13;
  LOBYTE(v22) = v14;
  v15 = (unsigned __int64)sub_CA1930(&v21) >> 1;
  switch ( (_DWORD)v15 )
  {
    case 1:
      LOWORD(v16) = 2;
      goto LABEL_13;
    case 2:
      LOWORD(v16) = 3;
LABEL_13:
      v18 = 0;
      goto LABEL_14;
    case 4:
      LOWORD(v16) = 4;
      goto LABEL_13;
    case 8:
      LOWORD(v16) = 5;
      goto LABEL_13;
    case 0x10:
      LOWORD(v16) = 6;
      goto LABEL_13;
    case 0x20:
      LOWORD(v16) = 7;
      goto LABEL_13;
    case 0x40:
      LOWORD(v16) = 8;
      goto LABEL_13;
    case 0x80:
      LOWORD(v16) = 9;
      goto LABEL_13;
  }
  v16 = sub_3007020(*(_QWORD **)(a1[1] + 64), v15);
  v6 = v16;
  v18 = v17;
LABEL_14:
  LOWORD(v6) = v16;
  sub_375B6E0(a1, a2, a3, v6, v18, a4, a6, v6, v18, a5);
}
