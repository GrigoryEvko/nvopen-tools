// Function: sub_324B240
// Address: 0x324b240
//
void __fastcall sub_324B240(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r14
  unsigned __int8 v7; // al
  __int64 v8; // rdi
  const void *v9; // rax
  size_t v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  unsigned __int64 v13; // r8
  unsigned int v14; // eax
  int v15; // eax
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int8 v19; // al
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int8 v22; // al
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int8 v25; // al
  __int64 v26; // r14
  unsigned __int64 v27; // rax
  char v28; // [rsp+Ch] [rbp-54h] BYREF
  __int64 v29; // [rsp+18h] [rbp-48h] BYREF
  __int64 v30[8]; // [rsp+20h] [rbp-40h] BYREF

  v4 = a3 - 16;
  v7 = *(_BYTE *)(a3 - 16);
  v28 = a4;
  if ( (v7 & 2) != 0 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 16LL);
    if ( !v8 )
      goto LABEL_25;
  }
  else
  {
    v8 = *(_QWORD *)(a3 - 8LL * ((v7 >> 2) & 0xF));
    if ( !v8 )
    {
LABEL_6:
      v11 = v4 - 8LL * ((v7 >> 2) & 0xF);
      goto LABEL_7;
    }
  }
  v9 = (const void *)sub_B91420(v8);
  if ( v10 )
    sub_324AD70(a1, a2, 3, v9, v10);
  v7 = *(_BYTE *)(a3 - 16);
  if ( (v7 & 2) == 0 )
    goto LABEL_6;
LABEL_25:
  v11 = *(_QWORD *)(a3 - 32);
LABEL_7:
  v12 = *(_QWORD *)(v11 + 24);
  if ( v12 )
    sub_32495E0(a1, a2, v12, 73);
  sub_3249E10(a1, a2, a3);
  v13 = *(_QWORD *)(a3 + 24);
  if ( v13 )
  {
    BYTE2(v30[0]) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 11, v30[0], v13 >> 3);
  }
  v14 = (unsigned int)sub_AF18D0(a3) >> 3;
  if ( v14 )
  {
    LODWORD(v30[0]) = 65551;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 136, 65551, v14 & 0x1FFFFFFF);
  }
  v15 = *(_DWORD *)(a3 + 20);
  if ( (v15 & 0x8000000) != 0 )
  {
    BYTE2(v30[0]) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 101, v30[0], 1);
  }
  else if ( (v15 & 0x10000000) != 0 )
  {
    BYTE2(v30[0]) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 101, v30[0], 2);
  }
  v30[0] = (__int64)a1;
  v29 = sub_3247B20((__int64)a1);
  v30[2] = (__int64)&v29;
  v30[3] = (__int64)&v28;
  v16 = *(_BYTE *)(a3 - 16);
  v30[1] = a2;
  if ( (v16 & 2) != 0 )
    v17 = *(_QWORD *)(a3 - 32);
  else
    v17 = v4 - 8LL * ((v16 >> 2) & 0xF);
  v18 = sub_AF2BB0(a3, *(_QWORD *)(v17 + 32));
  sub_324A3F0(v30, 34, v18);
  v19 = *(_BYTE *)(a3 - 16);
  if ( (v19 & 2) != 0 )
    v20 = *(_QWORD *)(a3 - 32);
  else
    v20 = v4 - 8LL * ((v19 >> 2) & 0xF);
  v21 = sub_AF2BB0(a3, *(_QWORD *)(v20 + 40));
  sub_324A3F0(v30, 47, v21);
  v22 = *(_BYTE *)(a3 - 16);
  if ( (v22 & 2) != 0 )
    v23 = *(_QWORD *)(a3 - 32);
  else
    v23 = v4 - 8LL * ((v22 >> 2) & 0xF);
  v24 = sub_AF2BB0(a3, *(_QWORD *)(v23 + 48));
  sub_324A3F0(v30, 46, v24);
  v25 = *(_BYTE *)(a3 - 16);
  if ( (v25 & 2) != 0 )
    v26 = *(_QWORD *)(a3 - 32);
  else
    v26 = v4 - 8LL * ((v25 >> 2) & 0xF);
  v27 = sub_AF2BB0(a3, *(_QWORD *)(v26 + 56));
  sub_324A3F0(v30, 8965, v27);
}
