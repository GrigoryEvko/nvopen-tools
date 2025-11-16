// Function: sub_2AC1F60
// Address: 0x2ac1f60
//
__int64 __fastcall sub_2AC1F60(__int64 a1, unsigned __int8 *a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  char v8; // r13
  char v9; // al
  __int64 v10; // rbx
  _QWORD *v11; // r15
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v20; // rsi
  int v21; // ecx
  unsigned __int8 *v22; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v23[10]; // [rsp+10h] [rbp-50h] BYREF

  v23[1] = &v22;
  v22 = a2;
  v23[0] = a1;
  v23[3] = sub_2AAE960;
  v23[2] = sub_2AA7D80;
  v8 = sub_2BF1270(v23, a5);
  sub_A17130((__int64)v23);
  v9 = sub_2AB37C0(*(_QWORD *)(a1 + 40), v22);
  if ( !v8
    && *(_BYTE *)(a5 + 4)
    && *v22 == 85
    && (v20 = *((_QWORD *)v22 - 4)) != 0
    && !*(_BYTE *)v20
    && *(_QWORD *)(v20 + 24) == *((_QWORD *)v22 + 10)
    && (*(_BYTE *)(v20 + 33) & 0x20) != 0
    && (v21 = *(_DWORD *)(v20 + 36), v8 = *(_BYTE *)(a5 + 4), v21 != 11) )
  {
    v8 = (unsigned int)(v21 - 210) <= 1;
    v10 = 0;
    if ( v9 )
      goto LABEL_19;
  }
  else
  {
    v10 = 0;
    if ( v9 )
LABEL_19:
      v10 = sub_2AB6F10(a1, *((_QWORD *)v22 + 5));
  }
  v11 = &a3[a4];
  v12 = sub_22077B0(0xA8u);
  v14 = v12;
  if ( v12 )
  {
    sub_2AC1B80(v12, 9, a3, v11, v22, v13);
    *(_BYTE *)(v14 + 160) = v8;
    *(_QWORD *)v14 = &unk_4A237B0;
    *(_QWORD *)(v14 + 40) = &unk_4A237F8;
    *(_QWORD *)(v14 + 96) = &unk_4A23830;
    *(_BYTE *)(v14 + 161) = v10 != 0;
    if ( v10 )
    {
      v17 = *(unsigned int *)(v14 + 56);
      if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v14 + 60) )
      {
        sub_C8D5F0(v14 + 48, (const void *)(v14 + 64), v17 + 1, 8u, v15, v16);
        v17 = *(unsigned int *)(v14 + 56);
      }
      *(_QWORD *)(*(_QWORD *)(v14 + 48) + 8 * v17) = v10;
      ++*(_DWORD *)(v14 + 56);
      v18 = *(unsigned int *)(v10 + 24);
      if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 28) )
      {
        sub_C8D5F0(v10 + 16, (const void *)(v10 + 32), v18 + 1, 8u, v15, v16);
        v18 = *(unsigned int *)(v10 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(v10 + 16) + 8 * v18) = v14 + 40;
      ++*(_DWORD *)(v10 + 24);
    }
  }
  return v14;
}
