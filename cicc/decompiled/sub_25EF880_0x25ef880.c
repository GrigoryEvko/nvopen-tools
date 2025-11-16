// Function: sub_25EF880
// Address: 0x25ef880
//
_QWORD *__fastcall sub_25EF880(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rbx
  int v10; // r12d
  __int64 v11; // rdi
  _QWORD *v12; // rsi
  _QWORD *v13; // rdx

  v5 = sub_B6AC80(a3, 358);
  v6 = sub_B6AC80(a3, 356);
  v7 = sub_B6AC80(a3, 357);
  if ( (v5 && *(_QWORD *)(v5 + 16) || v6 && *(_QWORD *)(v6 + 16) || v7 && *(_QWORD *)(v7 + 16))
    && (v8 = a3 + 8, v9 = *(_QWORD *)(a3 + 16), v10 = 0, v8 != v9) )
  {
    do
    {
      v11 = v9;
      v9 = *(_QWORD *)(v9 + 8);
      v10 |= sub_25EE6B0(v11 - 56);
    }
    while ( v8 != v9 );
    v12 = a1 + 4;
    v13 = a1 + 10;
    if ( (_BYTE)v10 )
    {
      memset(a1, 0, 0x60u);
      a1[1] = v12;
      *((_DWORD *)a1 + 4) = 2;
      *((_BYTE *)a1 + 28) = 1;
      a1[7] = v13;
      *((_DWORD *)a1 + 16) = 2;
      *((_BYTE *)a1 + 76) = 1;
      return a1;
    }
  }
  else
  {
    v12 = a1 + 4;
    v13 = a1 + 10;
  }
  a1[1] = v12;
  a1[2] = 0x100000002LL;
  a1[6] = 0;
  a1[4] = &qword_4F82400;
  a1[7] = v13;
  a1[8] = 2;
  *((_DWORD *)a1 + 18) = 0;
  *((_BYTE *)a1 + 76) = 1;
  *((_DWORD *)a1 + 6) = 0;
  *((_BYTE *)a1 + 28) = 1;
  *a1 = 1;
  return a1;
}
