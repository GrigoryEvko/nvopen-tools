// Function: sub_7FB010
// Address: 0x7fb010
//
_DWORD *__fastcall sub_7FB010(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rax
  int v7; // edx
  const __m128i *v8; // rdi

  v4 = *(_QWORD *)(a1 + 32);
  if ( dword_4F077C4 == 2 )
  {
    sub_733F40();
    sub_7E9A00(a1);
  }
  sub_7E1AA0();
  sub_72A130(a1);
  *(_BYTE *)(a1 + 29) |= 1u;
  sub_766370(a1);
  if ( !*(_BYTE *)(v4 + 172) && *(char *)(v4 + 192) >= 0 )
  {
    v5 = *(_QWORD *)(v4 + 272);
    if ( v5 )
    {
      v6 = *(_QWORD *)(v5 + 320);
      if ( !v6 )
        v6 = *(_QWORD *)(v4 + 272);
    }
    else
    {
      v6 = *(_QWORD *)(v4 + 320);
      if ( !v6 )
        goto LABEL_12;
    }
    if ( (*(_BYTE *)(v6 + 88) & 8) == 0 )
      goto LABEL_9;
LABEL_12:
    sub_7604D0(v4, 0xBu);
    sub_732AF0(v4);
    if ( dword_4F077C4 != 2 )
      goto LABEL_10;
LABEL_13:
    nullsub_13();
    goto LABEL_10;
  }
LABEL_9:
  sub_732AF0(v4);
  if ( dword_4F077C4 == 2 )
    goto LABEL_13;
LABEL_10:
  unk_4D03F40 = *(_QWORD *)(a3 + 120);
  sub_7E17F0(qword_4D03F60);
  v7 = *(unsigned __int8 *)(a3 + 104);
  v8 = *(const __m128i **)(a1 + 80);
  qword_4D03F60 = *(_QWORD **)(a3 + 112);
  dword_4D03EB8[0] = v7;
  sub_7FAF20(v8);
  unk_4D03EB0 = *(_QWORD *)(a3 + 176);
  qword_4F04C50 = *(_QWORD *)(a3 + 96);
  dword_4F04C58 = *(_DWORD *)(a3 + 92);
  sub_823780(a2);
  return sub_7296B0(*(_DWORD *)(a3 + 88));
}
