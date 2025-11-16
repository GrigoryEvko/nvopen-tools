// Function: sub_1A51850
// Address: 0x1a51850
//
unsigned __int64 *__fastcall sub_1A51850(unsigned __int64 *a1, __int64 a2, __int64 *a3)
{
  int v4; // eax
  __int64 v7; // rsi
  __int64 v8; // rcx
  int v9; // edx
  unsigned int v10; // eax
  _QWORD *v11; // r13
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  int v14; // r8d
  _QWORD v15[5]; // [rsp+8h] [rbp-78h] BYREF
  void *v16; // [rsp+30h] [rbp-50h]
  _QWORD v17[9]; // [rsp+38h] [rbp-48h] BYREF

  v4 = *(_DWORD *)(a2 + 24);
  if ( !v4 )
    goto LABEL_2;
  v7 = *(_QWORD *)(a2 + 8);
  v15[2] = -8;
  v15[3] = 0;
  v17[2] = -16;
  v8 = *a3;
  v9 = v4 - 1;
  v17[3] = 0;
  v15[0] = 2;
  v15[1] = 0;
  v17[0] = 2;
  v10 = (v4 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v17[1] = 0;
  v11 = (_QWORD *)(v7 + ((unsigned __int64)v10 << 6));
  v12 = v11[3];
  if ( v12 != v8 )
  {
    v14 = 1;
    while ( v12 != -8 )
    {
      v10 = v9 & (v14 + v10);
      v11 = (_QWORD *)(v7 + ((unsigned __int64)v10 << 6));
      v12 = v11[3];
      if ( v8 == v12 )
        goto LABEL_5;
      ++v14;
    }
    v16 = &unk_49EE2B0;
    sub_1455FA0((__int64)v17);
    sub_1455FA0((__int64)v15);
    goto LABEL_2;
  }
LABEL_5:
  v16 = &unk_49EE2B0;
  sub_1455FA0((__int64)v17);
  sub_1455FA0((__int64)v15);
  if ( v11 == (_QWORD *)(*(_QWORD *)(a2 + 8) + ((unsigned __int64)*(unsigned int *)(a2 + 24) << 6)) )
  {
LABEL_2:
    *a1 = 6;
    a1[1] = 0;
    a1[2] = 0;
    return a1;
  }
  *a1 = 6;
  v13 = v11[7];
  a1[1] = 0;
  a1[2] = v13;
  if ( v13 == -8 || v13 == 0 || v13 == -16 )
    return a1;
  sub_1649AC0(a1, v11[5] & 0xFFFFFFFFFFFFFFF8LL);
  return a1;
}
