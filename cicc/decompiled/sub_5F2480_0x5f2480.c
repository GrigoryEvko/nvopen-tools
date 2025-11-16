// Function: sub_5F2480
// Address: 0x5f2480
//
__int64 __fastcall sub_5F2480(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r12
  char v14; // al
  __int64 v15; // rsi
  __int64 v16; // r15
  int v17; // eax
  _BYTE *v18; // rax
  __int64 v19; // [rsp-10h] [rbp-90h]
  unsigned int v20; // [rsp+Ch] [rbp-74h] BYREF
  __int64 v21; // [rsp+10h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a1 + 88);
  v3 = *(_QWORD *)(a1 + 96);
  v4 = *(_QWORD *)(*(_QWORD *)(v2 + 168) + 152LL);
  sub_878710(a1, &v21);
  sub_87A530(&v21, 0);
  if ( !a2 || *a2 == v21 )
  {
    sub_7296C0(&v20);
    v6 = sub_72CBE0();
    v7 = sub_732700(v6, 0, 0, 0, 0, 0, 0, 0);
    v8 = *(_QWORD *)(v7 + 168);
    v9 = v7;
    *(_BYTE *)(v8 + 17) |= 2u;
    *(_BYTE *)(v8 + 21) |= 1u;
    v10 = dword_4D048B8;
    *(_QWORD *)(v8 + 40) = v2;
    v11 = v19;
    if ( (_DWORD)v10 )
    {
      v18 = (_BYTE *)sub_725E60(v6, v10, v19);
      *v18 |= 9u;
      *(_QWORD *)(v8 + 56) = v18;
    }
    v12 = sub_725FD0(v6, v10, v11);
    *(_QWORD *)(v12 + 152) = v9;
    v13 = v12;
    sub_725ED0(v12, 2);
    *(_DWORD *)(v13 + 192) |= 0x81000u;
    v14 = *(_BYTE *)(v13 + 195);
    if ( *(char *)(v2 + 177) < 0 )
    {
      v14 |= 8u;
      *(_BYTE *)(v13 + 195) = v14;
    }
    *(_BYTE *)(v13 + 195) = v14 | 0x10;
    v15 = v21;
    *(_QWORD *)(v13 + 112) = *(_QWORD *)(v4 + 144);
    *(_QWORD *)(v4 + 144) = v13;
    v16 = sub_87EBB0(10, v15);
    v17 = *(_DWORD *)(v4 + 24);
    *(_QWORD *)(v16 + 88) = v13;
    *(_DWORD *)(v16 + 40) = v17;
    *(_QWORD *)v13 = v16;
    sub_877D50(v13, *(_QWORD *)v16);
    sub_877E20(v16, v13, v2);
    *(_BYTE *)(v13 + 89) = *(_BYTE *)(v2 + 89) & 1 | *(_BYTE *)(v13 + 89) & 0xFE;
    sub_886160(v16);
    *(_QWORD *)(v3 + 24) = v16;
    sub_729730(v20);
    sub_736C90(v13, 1);
    if ( unk_4F068EC )
      sub_89A080(v13);
  }
  return *(_QWORD *)(v3 + 24);
}
