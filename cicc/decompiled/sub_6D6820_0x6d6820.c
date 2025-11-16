// Function: sub_6D6820
// Address: 0x6d6820
//
__int64 __fastcall sub_6D6820(int a1)
{
  __int64 v2; // r15
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rsi
  __m128i *v7; // r14
  __int64 v8; // r12
  __int64 v9; // rdi
  char v10; // al
  __int64 v11; // rdx
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-E8h]
  __int64 v18; // [rsp+8h] [rbp-E8h]
  char v19; // [rsp+13h] [rbp-DDh] BYREF
  unsigned int v20; // [rsp+14h] [rbp-DCh] BYREF
  __int64 *v21; // [rsp+18h] [rbp-D8h] BYREF
  _BYTE v22[18]; // [rsp+20h] [rbp-D0h] BYREF
  char v23; // [rsp+32h] [rbp-BEh]

  v2 = qword_4F06BC0;
  sub_6E1E00(2, v22, 0, 0);
  v23 |= 1u;
  sub_7296C0(&v20);
  qword_4F06BC0 = *(_QWORD *)(unk_4F07288 + 88LL);
  v5 = sub_6E2EF0(&v20, v22, v3, v4);
  v6 = 0;
  v7 = (__m128i *)(v5 + 8);
  v8 = v5;
  sub_69ED20(v5 + 8, 0, 0, 1);
  v9 = (__int64)v7;
  sub_68AD60(v7);
  v10 = *(_BYTE *)(v8 + 24);
  v11 = v8 + 152;
  if ( v10 != 2 )
  {
    if ( v10 != 1 )
      goto LABEL_3;
    v13 = *(_QWORD *)(v8 + 152);
    v14 = *(_BYTE *)(v13 + 24);
    if ( v14 == 2 )
    {
      v11 = *(_QWORD *)(v13 + 56);
    }
    else
    {
      if ( v14 != 3 )
        goto LABEL_3;
      v9 = *(_QWORD *)(*(_QWORD *)(v13 + 56) + 120LL);
      v17 = *(_QWORD *)(v13 + 56);
      if ( !(unsigned int)sub_8D32B0(v9) )
        goto LABEL_3;
      v15 = v17;
      if ( (*(_BYTE *)(v17 + 172) & 8) == 0 )
        goto LABEL_3;
      if ( (*(_BYTE *)(v17 + 89) & 2) != 0 )
      {
        v16 = sub_72F070(v17);
        v15 = v17;
        v6 = v16;
      }
      else
      {
        v6 = *(_QWORD *)(v17 + 40);
      }
      v9 = v15;
      sub_72F9F0(v15, v6, &v19, &v21);
      if ( v19 == 1 )
      {
        v11 = *v21;
      }
      else
      {
        if ( v19 != 2 || *(_BYTE *)(*v21 + 48) != 2 )
          goto LABEL_3;
        v11 = *(_QWORD *)(*v21 + 56);
      }
    }
    if ( !v11 )
      goto LABEL_3;
  }
  if ( *(_BYTE *)(v11 + 173) == 6 )
  {
    v9 = v11;
    v18 = v11;
    if ( !(unsigned int)sub_730990(v11) )
    {
      v6 = (__int64)v7;
      v9 = 2645;
      sub_6E6930(2645, v7, *(_QWORD *)(v18 + 128));
    }
  }
LABEL_3:
  unk_4D03C48 = 0;
  sub_6E2B30(v9, v6);
  *(_QWORD *)&dword_4F061D8 = *(_QWORD *)(v8 + 84);
  sub_729730(v20);
  qword_4F06BC0 = v2;
  if ( unk_4D03B6C != a1 )
    *(_BYTE *)(v8 + 28) |= 8u;
  return v8;
}
