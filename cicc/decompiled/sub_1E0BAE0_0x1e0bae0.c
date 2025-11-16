// Function: sub_1E0BAE0
// Address: 0x1e0bae0
//
__int64 __fastcall sub_1E0BAE0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  __int16 v7; // cx
  int v8; // r15d
  unsigned __int8 v9; // al
  int v10; // ebx
  unsigned int v11; // r13d
  __int64 v12; // r12
  char v14; // [rsp+8h] [rbp-78h]
  int v15; // [rsp+Ch] [rbp-74h]
  __int128 v16; // [rsp+10h] [rbp-70h]
  _QWORD v17[3]; // [rsp+30h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(a2 + 8) + a3;
  v6 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)a2 & 4) != 0 )
  {
    *((_QWORD *)&v16 + 1) = v5;
    *(_QWORD *)&v16 = v6 | 4;
  }
  else
  {
    if ( v6 )
      *(_QWORD *)&v16 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
    else
      *(_QWORD *)&v16 = 4;
    *((_QWORD *)&v16 + 1) = v5;
  }
  v7 = *(_WORD *)(a2 + 34);
  v8 = *(unsigned __int16 *)(a2 + 32);
  v9 = *(_BYTE *)(a2 + 37);
  v10 = *(unsigned __int8 *)(a2 + 36);
  memset(v17, 0, sizeof(v17));
  v11 = (unsigned int)(1 << v7) >> 1;
  v15 = v9 & 0xF;
  v14 = v9 >> 4;
  v12 = sub_145CBF0((__int64 *)(a1 + 120), 80, 16);
  sub_1E342C0(v12, v8, a4, v11, (unsigned int)v17, 0, v16, 0, v10, v15, v14);
  return v12;
}
