// Function: sub_7AD610
// Address: 0x7ad610
//
_QWORD *__fastcall sub_7AD610(unsigned __int64 a1)
{
  unsigned __int64 v1; // r13
  int v2; // ebx
  char *v3; // rax
  char *v4; // rcx
  char v5; // dl
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdx
  char v10; // al
  char v12; // [rsp+7h] [rbp-29h] BYREF
  _QWORD v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v13[0] = 0;
  v1 = (-(__int64)(a1 < 0x10000) & 0xFFFFFFFFFFFFFFFCLL) + 8;
  v2 = a1 < 0x10000 ? 4 : 8;
  v3 = (char *)v13 + v2;
  v4 = &v12 + v2 - (a1 < 0x10000 ? 3 : 7);
  do
  {
    v5 = a1;
    --v3;
    a1 >>= 4;
    *v3 = a0123456789abcd_0[v5 & 0xF];
  }
  while ( v4 != v3 );
  v6 = (_QWORD *)qword_4F17F68;
  v7 = *(_QWORD *)(qword_4F17F68 + 16);
  if ( (unsigned __int64)(v7 + 1) > *(_QWORD *)(qword_4F17F68 + 8) )
  {
    sub_823810(qword_4F17F68);
    v6 = (_QWORD *)qword_4F17F68;
    v7 = *(_QWORD *)(qword_4F17F68 + 16);
  }
  *(_BYTE *)(v6[4] + v7) = 92;
  v8 = v6[2];
  v9 = v8 + 1;
  v6[2] = v8 + 1;
  if ( (unsigned __int64)(v8 + 2) > v6[1] )
  {
    sub_823810(v6);
    v6 = (_QWORD *)qword_4F17F68;
    v9 = *(_QWORD *)(qword_4F17F68 + 16);
  }
  v10 = 117;
  if ( v2 == 8 )
    v10 = 85;
  *(_BYTE *)(v6[4] + v9) = v10;
  ++v6[2];
  sub_8238B0(v6, v13, v1);
  unk_4F072F3 = 1;
  return &qword_4F07280;
}
