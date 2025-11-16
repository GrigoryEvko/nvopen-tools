// Function: sub_18A8560
// Address: 0x18a8560
//
__int64 __fastcall sub_18A8560(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rdi
  const char *v6; // rax
  size_t v7; // rdx
  int *v8; // r15
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // r12
  int v12; // r14d
  __int64 v13; // rax
  int *v14; // r15
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // r14
  _QWORD *v20; // r13
  __int64 v21; // r12
  __int64 v22; // rbx
  __int64 v24; // rdi
  unsigned __int64 v25; // r13
  size_t v26; // [rsp+8h] [rbp-88h]
  unsigned int v27[2]; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v28[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v29[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v30[2]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v31[8]; // [rsp+50h] [rbp-40h] BYREF

  v3 = sub_15C70A0(a2 + 48);
  if ( !v3 )
    return 0;
  v4 = v3;
  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v5 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v5 + 16) )
    {
      v6 = sub_1649960(v5);
      v26 = v7;
      v8 = (int *)v6;
      v9 = sub_15C70A0(a2 + 48);
      if ( !v9 )
        goto LABEL_5;
LABEL_8:
      v10 = sub_393D1F0(*(_QWORD *)(a1 + 1200), v9);
      if ( !v10 )
        return 0;
      goto LABEL_9;
    }
  }
  v8 = 0;
  v26 = 0;
  v9 = sub_15C70A0(a2 + 48);
  if ( v9 )
    goto LABEL_8;
LABEL_5:
  v10 = *(_QWORD *)(a1 + 1200);
  if ( !v10 )
    return 0;
LABEL_9:
  LOBYTE(v29[0]) = 0;
  v28[0] = v29;
  v12 = 0;
  v13 = *(_QWORD *)(a1 + 1192);
  v28[1] = 0;
  v14 = sub_18A5420(v8, v26, *(_DWORD *)(v13 + 64), v28);
  v16 = v15;
  v17 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8));
  if ( *(_BYTE *)v17 == 19 )
  {
    v18 = *(_DWORD *)(v17 + 24);
    if ( (v18 & 1) == 0 )
    {
      v12 = (v18 >> 1) & 0x1F;
      if ( ((v18 >> 1) & 0x20) != 0 )
        v12 |= (v18 >> 2) & 0xFE0;
    }
  }
  v27[1] = v12;
  v27[0] = sub_393D1C0(v4);
  v19 = sub_18A83F0(v10 + 80, v27);
  if ( v19 != v10 + 88 )
  {
    if ( v14 )
    {
      v30[0] = (__int64)v31;
      sub_18A3750(v30, v14, (__int64)v14 + v16);
      v20 = (_QWORD *)v30[0];
      v21 = sub_18A8460(v19 + 40, (__int64)v30);
      if ( v20 != v31 )
        j_j___libc_free_0(v20, v31[0] + 1LL);
      v22 = v19 + 48;
      if ( v21 != v19 + 48 )
        goto LABEL_16;
    }
    else
    {
      LOBYTE(v31[0]) = 0;
      v30[0] = (__int64)v31;
      v22 = v19 + 48;
      v30[1] = 0;
      v21 = sub_18A8460(v19 + 40, (__int64)v30);
      if ( v21 != v19 + 48 )
      {
LABEL_16:
        v11 = v21 + 64;
        goto LABEL_17;
      }
    }
    v24 = *(_QWORD *)(v19 + 64);
    if ( v22 != v24 )
    {
      v11 = 0;
      v25 = 0;
      do
      {
        if ( *(_QWORD *)(v24 + 80) >= v25 )
        {
          v11 = v24 + 64;
          v25 = *(_QWORD *)(v24 + 80);
        }
        v24 = sub_220EF30(v24);
      }
      while ( v22 != v24 );
      goto LABEL_17;
    }
  }
  v11 = 0;
LABEL_17:
  if ( (_QWORD *)v28[0] != v29 )
    j_j___libc_free_0(v28[0], v29[0] + 1LL);
  return v11;
}
