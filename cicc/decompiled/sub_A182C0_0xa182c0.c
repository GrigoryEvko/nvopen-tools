// Function: sub_A182C0
// Address: 0xa182c0
//
void __fastcall sub_A182C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  char v7; // r8
  _QWORD *v8; // rax
  char v9; // al
  __int64 v10; // rdi

  v2 = (unsigned int)qword_4F807E8;
  v3 = a2;
  v5 = (_QWORD *)sub_22077B0(152);
  v6 = v5;
  if ( v5 )
  {
    v5[1] = 0;
    *v5 = v5 + 3;
    v5[2] = 0;
    v7 = sub_CB7450(a2);
    v8 = v6;
    if ( v7 )
      v8 = *(_QWORD **)(a2 + 48);
    v6[3] = v8;
    v9 = sub_CB7450(a2);
    v6[6] = 0;
    v6[7] = 2;
    if ( v9 )
      v3 = 0;
    v6[8] = 0;
    v6[5] = v2 << 20;
    v6[4] = v3;
    v6[9] = 0;
    v6[10] = 0;
    *((_BYTE *)v6 + 96) = 0;
    v6[13] = 0;
    v6[14] = 0;
    v6[15] = 0;
    v6[16] = 0;
    v6[17] = 0;
    v6[18] = 0;
  }
  *(_QWORD *)a1 = v6;
  sub_C0BFB0(a1 + 8, 6, 0);
  v10 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x400000000LL;
  *(_QWORD *)(a1 + 120) = a1 + 136;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 1;
  *(_WORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  sub_A17BC0(v10);
}
