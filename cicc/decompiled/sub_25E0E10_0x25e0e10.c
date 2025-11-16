// Function: sub_25E0E10
// Address: 0x25e0e10
//
_BYTE *__fastcall sub_25E0E10(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  _BYTE *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  char *v12; // r12
  unsigned __int64 v13; // rsi
  __int64 v14; // r9
  unsigned __int64 v15; // rdx
  unsigned __int64 *v16; // rdi
  unsigned __int64 v17; // rdx
  _BYTE *result; // rax
  char *v19; // r12
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  _QWORD v25[2]; // [rsp+10h] [rbp-50h] BYREF
  _BYTE *v26; // [rsp+20h] [rbp-40h]

  v6 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v7 = *(_QWORD **)(a2 - 8);
    v8 = &v7[v6];
  }
  else
  {
    v8 = (_QWORD *)a2;
    v7 = (_QWORD *)(a2 - v6 * 8);
  }
  for ( ; v8 != v7; v7 += 4 )
  {
    v9 = (_BYTE *)*v7;
    if ( *(_BYTE *)*v7 > 0x1Cu )
    {
      v25[0] = 6;
      v10 = *a1;
      v25[1] = 0;
      v26 = v9;
      if ( v9 != (_BYTE *)-8192LL && v9 != (_BYTE *)-4096LL )
      {
        v21 = v10;
        sub_BD73F0((__int64)v25);
        v10 = v21;
      }
      v11 = *(unsigned int *)(v10 + 8);
      v12 = (char *)v25;
      v13 = *(_QWORD *)v10;
      v14 = v11 + 1;
      v15 = v11;
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 12) )
      {
        if ( v13 > (unsigned __int64)v25 || (v15 = v13 + 24 * v11, (unsigned __int64)v25 >= v15) )
        {
          v24 = v10;
          sub_F39130(v10, v11 + 1, v15, v11, a5, v14);
          v10 = v24;
          v11 = *(unsigned int *)(v24 + 8);
          v13 = *(_QWORD *)v24;
          LODWORD(v15) = *(_DWORD *)(v24 + 8);
        }
        else
        {
          v19 = (char *)v25 - v13;
          v23 = v10;
          sub_F39130(v10, v11 + 1, v15, v11, a5, v14);
          v10 = v23;
          v13 = *(_QWORD *)v23;
          v11 = *(unsigned int *)(v23 + 8);
          v12 = &v19[*(_QWORD *)v23];
          LODWORD(v15) = *(_DWORD *)(v23 + 8);
        }
      }
      v16 = (unsigned __int64 *)(v13 + 24 * v11);
      if ( v16 )
      {
        *v16 = 6;
        v17 = *((_QWORD *)v12 + 2);
        v16[1] = 0;
        v16[2] = v17;
        if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
        {
          v22 = v10;
          sub_BD6050(v16, *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL);
          v10 = v22;
        }
        LODWORD(v15) = *(_DWORD *)(v10 + 8);
      }
      *(_DWORD *)(v10 + 8) = v15 + 1;
      if ( v26 + 4096 != 0 && v26 != 0 && v26 != (_BYTE *)-8192LL )
        sub_BD60C0(v25);
    }
  }
  sub_B43D60((_QWORD *)a2);
  result = (_BYTE *)a1[1];
  *result = 1;
  return result;
}
