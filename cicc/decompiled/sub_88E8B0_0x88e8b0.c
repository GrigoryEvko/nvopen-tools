// Function: sub_88E8B0
// Address: 0x88e8b0
//
_QWORD *__fastcall sub_88E8B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  _QWORD *v3; // r14
  __m128i *v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // rax
  _QWORD *v7; // r15
  char v8; // dl
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v16; // rax
  __m128i *v17; // rax
  _QWORD *v18; // rax
  int v19[13]; // [rsp+1Ch] [rbp-34h] BYREF

  sub_7296C0(v19);
  v2 = *(_QWORD *)(a1 + 88);
  v3 = sub_87F360(a1);
  if ( (*((_BYTE *)v3 + 81) & 0x10) == 0
    && dword_4F077C4 == 2
    && (unk_4F07778 > 201102 || dword_4F07774)
    && (v16 = v3[8]) != 0
    && (*(_BYTE *)(v16 + 124) & 4) != 0 )
  {
    v17 = sub_725D10(2);
    v17[10].m128i_i8[10] |= 0x10u;
    v5 = (__int64)v17;
    v18 = sub_725CE0();
    *(_QWORD *)(v5 + 216) = v18;
    v7 = v18;
    *(_BYTE *)(v5 + 88) = *(_BYTE *)(v5 + 88) & 0x8F | 0x10;
  }
  else
  {
    v4 = sub_725D10(1);
    v4[10].m128i_i8[10] |= 0x10u;
    v5 = (__int64)v4;
    v6 = sub_725CE0();
    *(_QWORD *)(v5 + 216) = v6;
    v7 = v6;
    *(_BYTE *)(v5 + 88) = *(_BYTE *)(v5 + 88) & 0x8F | 0x20;
  }
  *(_BYTE *)(v5 + 157) = *(_BYTE *)(*(_QWORD *)(v2 + 104) + 184LL) & 1 | *(_BYTE *)(v5 + 157) & 0xFE;
  v8 = *(_BYTE *)(*(_QWORD *)(v2 + 104) + 184LL) & 4 | *(_BYTE *)(v5 + 156) & 0xFB;
  *(_BYTE *)(v5 + 156) = v8;
  v9 = v8 & 0xFE | ((*(_BYTE *)(*(_QWORD *)(v2 + 104) + 184LL) & 0x10) != 0);
  *(_BYTE *)(v5 + 156) = v9;
  *(_BYTE *)(v5 + 156) = *(_BYTE *)(*(_QWORD *)(v2 + 104) + 184LL) & 2 | v9 & 0xFD;
  v3[11] = v5;
  v7[2] = *(_QWORD *)(v2 + 104);
  *(_BYTE *)(v5 + 88) = sub_87D550(a1) & 3 | *(_BYTE *)(v5 + 88) & 0xFC;
  v10 = sub_880C60();
  *(_QWORD *)(v10 + 32) = a1;
  *(_QWORD *)(v10 + 24) = v3;
  v3[12] = v10;
  *v7 = a2;
  *(_QWORD *)(*(_QWORD *)(v5 + 216) + 16LL) = *(_QWORD *)(v2 + 104);
  sub_877D80(v5, v3);
  sub_877F10(v5, (__int64)v3, v11, v12, v13, v14);
  sub_729730(v19[0]);
  return v3;
}
