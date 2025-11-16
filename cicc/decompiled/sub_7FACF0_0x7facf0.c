// Function: sub_7FACF0
// Address: 0x7facf0
//
__int64 __fastcall sub_7FACF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __m128i *v4; // r14
  _BYTE *v5; // rax
  char v6; // r11
  _BYTE *v7; // r10
  _BYTE *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rax
  __m128i *v13; // rax
  _BYTE *v14; // rax
  __int64 v15; // [rsp+8h] [rbp-78h]
  unsigned __int8 v16; // [rsp+10h] [rbp-70h]
  int v17; // [rsp+14h] [rbp-6Ch]
  __int16 v18; // [rsp+18h] [rbp-68h]
  unsigned __int16 v19; // [rsp+1Ah] [rbp-66h]
  int v20; // [rsp+1Ch] [rbp-64h]
  __int64 v21; // [rsp+20h] [rbp-60h] BYREF
  __int64 v22; // [rsp+28h] [rbp-58h] BYREF
  __m128i v23[5]; // [rsp+30h] [rbp-50h] BYREF

  v20 = dword_4D03F38[0];
  v19 = dword_4D03F38[1];
  v17 = dword_4F07508[0];
  v18 = dword_4F07508[1];
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 64);
  *(_QWORD *)dword_4D03F38 = *(_QWORD *)dword_4F07508;
  if ( (*(_BYTE *)(a2 - 8) & 1) != 0 && !*(_QWORD *)(a2 - 16) )
    sub_729FB0(a2, 2, qword_4D03FF0);
  *(_BYTE *)(a1 + 177) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  if ( *(_BYTE *)(a2 + 173) == 10 || (unsigned int)sub_8D3410(*(_QWORD *)(a1 + 120)) )
  {
    v3 = *(_QWORD *)(a1 + 120);
    if ( (*(_BYTE *)(a2 - 8) & 1) != 0 )
    {
      v13 = sub_7E9240(v3);
      v13[11].m128i_i8[1] = 1;
      v4 = v13;
      v13[11].m128i_i64[1] = a2;
    }
    else
    {
      v4 = sub_7E9300(v3, 1);
      sub_7333B0((__int64)v4, (_BYTE *)qword_4D03F68[1], 1, a2, 0);
    }
    v5 = sub_731250((__int64)v4);
    v6 = 86;
    v7 = v5;
  }
  else
  {
    v14 = sub_7EBB70(a2);
    v6 = 73;
    v7 = v14;
  }
  v16 = v6;
  v15 = (__int64)v7;
  sub_7E1740(*(_QWORD *)(qword_4F04C50 + 80LL), (__int64)v23);
  sub_7F8CF0(a1, v23[0].m128i_i32, v23, &v21, &v22);
  v8 = sub_731250(a1);
  v11 = sub_7E6A80(v8, v16, v15, v23[0].m128i_i32, v9, v10);
  if ( v11 )
  {
    *v11 = *(_QWORD *)dword_4D03F38;
    v11[1] = *(_QWORD *)dword_4D03F38;
  }
  *(_BYTE *)(a1 + 173) |= 8u;
  dword_4F07508[0] = v17;
  LOWORD(dword_4F07508[1]) = v18;
  dword_4D03F38[0] = v20;
  LOWORD(dword_4D03F38[1]) = v19;
  return v19;
}
