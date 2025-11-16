// Function: sub_25C3660
// Address: 0x25c3660
//
void __fastcall sub_25C3660(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 i; // r12
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rsi
  int v13; // edx
  __int64 v14; // r15
  __int64 v15; // rdi
  __int64 v16; // r14
  __int64 v17; // r9
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v25; // [rsp+20h] [rbp-90h]
  char v26; // [rsp+28h] [rbp-88h]
  __int64 v27[2]; // [rsp+30h] [rbp-80h] BYREF
  char v28; // [rsp+40h] [rbp-70h] BYREF

  v4 = a1 + 96 * a2;
  v5 = (a3 - 1) / 2;
  v6 = v4 + 16;
  if ( a2 >= v5 )
  {
    v8 = a2;
    v10 = v4 + 16;
  }
  else
  {
    for ( i = a2; ; i = v8 )
    {
      v8 = 2 * (i + 1);
      v4 = a1 + 192 * (i + 1);
      if ( sub_B445A0(*(_QWORD *)v4, *(_QWORD *)(v4 - 96)) )
      {
        --v8;
        v4 = a1 + 96 * v8;
      }
      v9 = 3 * i;
      v10 = v4 + 16;
      v11 = a1 + 32 * v9;
      *(_QWORD *)v11 = *(_QWORD *)v4;
      *(_BYTE *)(v11 + 8) = *(_BYTE *)(v4 + 8);
      sub_25C2C90(v6, (__int64 *)(v4 + 16));
      if ( v8 >= v5 )
        break;
      v6 = v4 + 16;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v8 )
  {
    v17 = v8 + 1;
    v18 = 2 * (v8 + 1);
    v19 = v18 + 4 * v17;
    v8 = v18 - 1;
    v20 = a1 + 32 * v19 - 96;
    *(_QWORD *)v4 = *(_QWORD *)v20;
    *(_BYTE *)(v4 + 8) = *(_BYTE *)(v20 + 8);
    sub_25C2C90(v10, (__int64 *)(v20 + 16));
    v4 = a1 + 96 * v8;
    v10 = v4 + 16;
  }
  v12 = *(_QWORD *)a4;
  v13 = *(_DWORD *)(a4 + 24);
  v26 = *(_BYTE *)(a4 + 8);
  v27[0] = (__int64)&v28;
  v27[1] = 0x200000000LL;
  v25 = v12;
  if ( v13 )
    sub_25C2C90((__int64)v27, (__int64 *)(a4 + 16));
  v14 = (v8 - 1) / 2;
  if ( v8 > a2 )
  {
    while ( 1 )
    {
      v16 = a1 + 96 * v14;
      v4 = 96 * v8 + a1;
      if ( !sub_B445A0(*(_QWORD *)v16, v12) )
      {
        v12 = v25;
        goto LABEL_18;
      }
      v15 = v10;
      *(_QWORD *)v4 = *(_QWORD *)v16;
      *(_BYTE *)(v4 + 8) = *(_BYTE *)(v16 + 8);
      v10 = v16 + 16;
      sub_25C2C90(v15, (__int64 *)(v16 + 16));
      v12 = v25;
      if ( a2 >= v14 )
        break;
      v8 = v14;
      v14 = (v14 - 1) / 2;
    }
    v4 = a1 + 96 * v14;
  }
LABEL_18:
  *(_QWORD *)v4 = v12;
  *(_BYTE *)(v4 + 8) = v26;
  sub_25C2C90(v10, v27);
  sub_25C0430((__int64)v27);
}
