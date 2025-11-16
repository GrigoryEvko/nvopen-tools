// Function: sub_14A4F70
// Address: 0x14a4f70
//
__int64 __fastcall sub_14A4F70(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rdi
  __int64 (*v3)(void); // rax
  __int64 *v4; // r13
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // r12
  _BYTE *v11; // rdx
  int v12; // ecx
  _BYTE *v13; // r9
  __int64 *v14; // rax
  __int64 v15; // rcx
  unsigned __int8 v16; // al
  unsigned int v17; // ebx
  int v18; // eax
  char v19; // al
  __int64 v20; // rbx
  _BYTE *v21; // rsi
  char v22; // al
  unsigned int v23; // r12d
  __int64 v24; // rbx
  __int64 **v25; // rax
  __int64 *v26; // r12
  __int64 v27; // rax
  char v29; // al
  __int64 v30; // [rsp+0h] [rbp-70h]
  __int64 v31; // [rsp+8h] [rbp-68h]
  unsigned __int64 v32; // [rsp+8h] [rbp-68h]
  _BYTE *v33; // [rsp+8h] [rbp-68h]
  _BYTE *v34; // [rsp+10h] [rbp-60h] BYREF
  __int64 v35; // [rsp+18h] [rbp-58h]
  _BYTE v36[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = *a1;
  v3 = *(__int64 (**)(void))(*v2 + 872);
  if ( v3 != sub_14A6690 )
    return v3();
  v4 = v2 + 1;
  v5 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v6 = *(__int64 **)(a2 - 8);
    v7 = (__int64)&v6[v5];
  }
  else
  {
    v6 = (__int64 *)(a2 - v5 * 8);
    v7 = a2;
  }
  v8 = v7 - (_QWORD)v6;
  v34 = v36;
  v35 = 0x400000000LL;
  v9 = 0xAAAAAAAAAAAAAAABLL * (v8 >> 3);
  v10 = v9;
  if ( (unsigned __int64)v8 > 0x60 )
  {
    v30 = v8;
    v32 = 0xAAAAAAAAAAAAAAABLL * (v8 >> 3);
    sub_16CD150(&v34, v36, v32, 8);
    v13 = v34;
    v12 = v35;
    LODWORD(v9) = v32;
    v8 = v30;
    v11 = &v34[8 * (unsigned int)v35];
  }
  else
  {
    v11 = v36;
    v12 = 0;
    v13 = v36;
  }
  if ( v8 > 0 )
  {
    v14 = v6;
    do
    {
      v15 = *v14;
      v11 += 8;
      v14 += 3;
      *((_QWORD *)v11 - 1) = v15;
      --v10;
    }
    while ( v10 );
    v13 = v34;
    v12 = v35;
  }
  v16 = *(_BYTE *)(a2 + 16);
  v17 = v9 + v12;
  LODWORD(v35) = v9 + v12;
  if ( v16 <= 0x17u )
    goto LABEL_12;
  if ( v16 == 77 || v16 == 86 )
    goto LABEL_34;
  if ( v16 != 53 )
    goto LABEL_29;
  v33 = v13;
  v29 = sub_15F8F00(a2);
  v13 = v33;
  if ( v29 )
    goto LABEL_33;
  v16 = *(_BYTE *)(a2 + 16);
  if ( v16 <= 0x17u )
  {
LABEL_12:
    if ( v16 != 5 || *(_WORD *)(a2 + 18) != 32 )
      goto LABEL_13;
  }
  else
  {
LABEL_29:
    if ( v16 != 56 )
    {
LABEL_13:
      v18 = sub_14A4C90((__int64)v4, a2);
      goto LABEL_14;
    }
  }
  v24 = v17 - 1LL;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v25 = *(__int64 ***)(a2 - 8);
  else
    v25 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v26 = *v25;
  v31 = (__int64)(v13 + 8);
  v27 = sub_16348C0(a2);
  v18 = sub_14A1310(v4, v27, v26, v31, v24);
LABEL_14:
  if ( !v18 )
  {
LABEL_33:
    v13 = v34;
LABEL_34:
    v23 = 0;
    goto LABEL_35;
  }
  v19 = *(_BYTE *)(a2 + 16);
  if ( v19 == 54 )
  {
    v13 = v34;
    v23 = 4;
  }
  else
  {
    v20 = *(_QWORD *)a2;
    if ( v19 != 78 )
      goto LABEL_21;
    v21 = *(_BYTE **)(a2 - 24);
    if ( !v21[16] && !sub_14A2090((__int64)v4, v21) )
    {
      v22 = *(_BYTE *)(v20 + 8);
      if ( v22 != 13 )
      {
LABEL_22:
        if ( v22 == 16 )
          v22 = *(_BYTE *)(*(_QWORD *)(v20 + 24) + 8LL);
        v13 = v34;
        v23 = (unsigned __int8)(v22 - 1) < 6u ? 3 : 1;
        goto LABEL_35;
      }
      v20 = **(_QWORD **)(v20 + 16);
LABEL_21:
      v22 = *(_BYTE *)(v20 + 8);
      goto LABEL_22;
    }
    v13 = v34;
    v23 = 40;
  }
LABEL_35:
  if ( v13 != v36 )
    _libc_free((unsigned __int64)v13);
  return v23;
}
