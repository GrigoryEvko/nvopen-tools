// Function: sub_2128150
// Address: 0x2128150
//
__int64 __fastcall sub_2128150(__int64 *a1, unsigned __int64 a2)
{
  __int64 v4; // rbx
  unsigned int v5; // ecx
  unsigned __int8 v6; // al
  int v7; // eax
  char v8; // r14
  __int64 v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // rdx
  _QWORD *v12; // rdi
  __int64 v13; // r14
  const __m128i *v14; // r9
  unsigned int v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  int v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v17,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = v19;
  v5 = (unsigned __int8)v18;
  v6 = *(_BYTE *)(a2 + 27);
  if ( *(_WORD *)(a2 + 24) != 185 )
  {
    LOBYTE(v7) = (v6 >> 2) & 3;
LABEL_3:
    v8 = v7;
    goto LABEL_4;
  }
  v8 = 1;
  v7 = (v6 >> 2) & 3;
  if ( v7 )
    goto LABEL_3;
LABEL_4:
  v9 = *(_QWORD *)(a2 + 72);
  v17 = v9;
  if ( v9 )
  {
    v16 = (unsigned __int8)v18;
    sub_1623A60((__int64)&v17, v9, 2);
    v5 = v16;
  }
  v10 = *(_QWORD *)(a2 + 32);
  v11 = *(_QWORD *)(a2 + 96);
  v12 = (_QWORD *)a1[1];
  v18 = *(_DWORD *)(a2 + 64);
  v13 = sub_1D2B590(
          v12,
          v8,
          (__int64)&v17,
          v5,
          v4,
          *(_QWORD *)(a2 + 104),
          *(_OWORD *)v10,
          *(_QWORD *)(v10 + 40),
          *(_QWORD *)(v10 + 48),
          *(unsigned __int8 *)(a2 + 88),
          v11);
  sub_2013400((__int64)a1, a2, 1, v13, (__m128i *)1, v14);
  if ( v17 )
    sub_161E7C0((__int64)&v17, v17);
  return v13;
}
