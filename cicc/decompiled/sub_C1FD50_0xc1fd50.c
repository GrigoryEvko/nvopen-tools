// Function: sub_C1FD50
// Address: 0xc1fd50
//
__int64 __fastcall sub_C1FD50(__int64 a1, _DWORD *a2)
{
  __int64 v4; // rdx
  _BYTE *v5; // rsi
  __int64 v6; // rdi
  int v7; // eax
  int v8; // edx
  int v9; // eax
  unsigned int v10; // r12d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  char v15; // al
  char v16; // dl
  char *v17; // [rsp+0h] [rbp-40h] BYREF
  __int64 v18; // [rsp+8h] [rbp-38h]
  _QWORD v19[6]; // [rsp+10h] [rbp-30h] BYREF

  v17 = (char *)v19;
  v5 = (_BYTE *)sub_C5ED50(a1, a1 + 24, 4, a1 + 32);
  sub_C1EB20((__int64 *)&v17, v5, (__int64)&v5[v4]);
  v6 = (__int64)v17;
  if ( v18 != 4 )
  {
LABEL_12:
    v10 = 0;
    goto LABEL_13;
  }
  if ( *(_BYTE *)(a1 + 16) && v17 < v17 + 3 )
  {
    v15 = *v17;
    v16 = v17[3];
    *(_WORD *)(v17 + 1) = __ROL2__(*(_WORD *)(v17 + 1), 8);
    *(_BYTE *)v6 = v16;
    *(_BYTE *)(v6 + 3) = v15;
    v6 = (__int64)v17;
    v7 = *v17;
    v8 = v17[2];
    if ( *v17 > 64 )
    {
LABEL_16:
      v9 = v8 + 100 * (char)(v7 - 65) + 2 * (5 * *(char *)(v6 + 1) - 240) - 48;
      if ( v9 > 119 )
        goto LABEL_17;
      goto LABEL_5;
    }
  }
  else
  {
    v7 = *v17;
    v8 = v17[2];
    if ( *v17 > 64 )
      goto LABEL_16;
  }
  v9 = v8 + 2 * (5 * v7 - 240) - 48;
  if ( v9 > 119 )
  {
LABEL_17:
    *a2 = 5;
    v10 = 1;
    *(_DWORD *)(a1 + 48) = 5;
    goto LABEL_13;
  }
LABEL_5:
  if ( v9 > 89 )
  {
    *a2 = 4;
    v10 = 1;
    *(_DWORD *)(a1 + 48) = 4;
  }
  else if ( v9 > 79 )
  {
    *a2 = 3;
    v10 = 1;
    *(_DWORD *)(a1 + 48) = 3;
  }
  else if ( v9 > 47 )
  {
    *a2 = 2;
    v10 = 1;
    *(_DWORD *)(a1 + 48) = 2;
  }
  else
  {
    if ( v9 != 47 )
    {
      if ( v9 > 33 )
      {
        *a2 = 0;
        v10 = 1;
        *(_DWORD *)(a1 + 48) = 0;
        goto LABEL_13;
      }
      v11 = sub_CB72A0(v6, v5);
      v12 = sub_904010(v11, "unexpected version: ");
      v13 = sub_CB6200(v12, v17, v18);
      sub_904010(v13, "\n");
      v6 = (__int64)v17;
      goto LABEL_12;
    }
    *a2 = 1;
    v10 = 1;
    *(_DWORD *)(a1 + 48) = 1;
  }
LABEL_13:
  if ( (_QWORD *)v6 != v19 )
    j_j___libc_free_0(v6, v19[0] + 1LL);
  return v10;
}
