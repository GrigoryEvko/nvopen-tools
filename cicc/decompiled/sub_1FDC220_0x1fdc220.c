// Function: sub_1FDC220
// Address: 0x1fdc220
//
__int64 __fastcall sub_1FDC220(__int64 **a1, __int64 a2, unsigned int a3)
{
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int64 v8; // rsi
  unsigned int v9; // ebx
  unsigned int v10; // r15d
  __int64 *v11; // rax
  __int64 v12; // rbx
  int v13; // ecx
  unsigned int v14; // ecx
  unsigned int v15; // ebx
  __int64 *v16; // rax
  unsigned __int8 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned int v20; // eax
  unsigned __int64 *v21; // r9
  unsigned __int64 v22; // r9
  unsigned int v23; // eax
  __int64 v24; // rax
  unsigned int v25; // edx
  __int64 v26; // rax
  bool v27; // al
  __int64 (*v28)(); // r10
  unsigned int v29; // edx
  unsigned int v30; // ecx
  __int64 v31; // rax
  unsigned __int8 v32; // al
  unsigned __int64 v33; // r9
  bool v34; // al
  unsigned __int64 v35; // rax
  __int64 v36; // [rsp+0h] [rbp-60h]
  unsigned int v37; // [rsp+0h] [rbp-60h]
  unsigned __int8 v38; // [rsp+0h] [rbp-60h]
  unsigned __int8 v39; // [rsp+8h] [rbp-58h]
  unsigned int v40; // [rsp+8h] [rbp-58h]
  unsigned __int64 v41; // [rsp+8h] [rbp-58h]
  _BYTE v42[80]; // [rsp+10h] [rbp-50h] BYREF

  LOBYTE(v5) = sub_1F59570(*(_QWORD *)a2);
  if ( (unsigned __int8)v5 <= 1u )
    return 0;
  v8 = (__int64)a1[14];
  v9 = v5;
  v10 = v5;
  if ( !*(_QWORD *)(v8 + 8LL * (unsigned __int8)v5 + 120) )
  {
    v36 = v6;
    if ( (_BYTE)v5 != 2 || a3 - 118 > 2 )
      return 0;
    LOBYTE(v9) = 2;
    v24 = sub_16498A0(a2);
    sub_1F40D10((__int64)v42, v8, v24, v9, v36);
    v10 = v42[8];
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v11 = *(__int64 **)(a2 - 8);
  else
    v11 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v12 = *v11;
  if ( *(_BYTE *)(*v11 + 16) == 13 )
  {
    v13 = *(unsigned __int8 *)(a2 + 16);
    if ( (unsigned __int8)v13 > 0x17u )
    {
      v14 = v13 - 24;
      if ( v14 <= 0x1C && ((1LL << v14) & 0x1C019800) != 0 )
      {
        v30 = sub_1FD8F60(a1, v11[3]);
        if ( !v30 )
          return 0;
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v31 = *(_QWORD *)(a2 - 8);
        else
          v31 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v40 = v30;
        v32 = sub_1FD4DC0((__int64)a1, *(_QWORD *)(v31 + 24));
        if ( *(_DWORD *)(v12 + 32) <= 0x40u )
          v33 = *(_QWORD *)(v12 + 24);
        else
          v33 = **(_QWORD **)(v12 + 24);
        v23 = sub_1FDC040(a1, v10, a3, v40, v32, v33, v10);
        goto LABEL_25;
      }
    }
  }
  v15 = sub_1FD8F60(a1, *v11);
  if ( !v15 )
    return 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v16 = *(__int64 **)(a2 - 8);
  else
    v16 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v17 = sub_1FD4DC0((__int64)a1, *v16);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v18 = *(_QWORD *)(a2 - 8);
  else
    v18 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v19 = *(_QWORD *)(v18 + 24);
  if ( *(_BYTE *)(v19 + 16) == 13 )
  {
    v20 = *(_DWORD *)(v19 + 32);
    v21 = *(unsigned __int64 **)(v19 + 24);
    if ( v20 <= 0x40 )
      v22 = (__int64)((_QWORD)v21 << (64 - (unsigned __int8)v20)) >> (64 - (unsigned __int8)v20);
    else
      v22 = *v21;
    if ( a3 == 55 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(a2 + 16) - 35) <= 0x11u )
      {
        v38 = v17;
        v41 = v22;
        v34 = sub_15F23D0(a2);
        v22 = v41;
        v17 = v38;
        if ( v41 )
        {
          if ( v34 && (v41 & (v41 - 1)) == 0 )
          {
            _BitScanReverse64(&v35, v41);
            a3 = 123;
            v22 = 63 - ((unsigned int)v35 ^ 0x3F);
          }
        }
      }
    }
    else if ( a3 == 58 && (unsigned __int8)(*(_BYTE *)(a2 + 16) - 35) <= 0x11u && v22 && ((v22 - 1) & v22) == 0 )
    {
      --v22;
      a3 = 118;
    }
    v23 = sub_1FDC040(a1, v10, a3, v15, v17, v22, v10);
LABEL_25:
    if ( v23 )
    {
      sub_1FD5CC0((__int64)a1, a2, v23, 1);
      return 1;
    }
    return 0;
  }
  v39 = v17;
  v25 = sub_1FD8F60(a1, v19);
  if ( !v25 )
    return 0;
  v26 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v37 = v25;
  v27 = sub_1FD4DC0((__int64)a1, *(_QWORD *)(v26 + 24));
  v28 = (__int64 (*)())(*a1)[9];
  if ( v28 == sub_1FD34D0 )
    return 0;
  v29 = ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, bool))v28)(
          a1,
          v10,
          v10,
          a3,
          v15,
          v39,
          v37,
          v27);
  if ( !v29 )
    return 0;
  sub_1FD5CC0((__int64)a1, a2, v29, 1);
  return 1;
}
