// Function: sub_1155C30
// Address: 0x1155c30
//
__int64 __fastcall sub_1155C30(_BYTE *a1, __int64 *a2, __int64 a3, bool *a4)
{
  __int64 v7; // rax
  char v8; // cl
  char v9; // dl
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int8 *v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rbx
  unsigned int v16; // eax
  bool v17; // cc
  unsigned int v18; // r12d
  unsigned __int64 v19; // r14
  _QWORD *v20; // rax
  bool v21; // al
  __int64 v22; // rdx
  unsigned __int8 *v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rdx
  _BYTE *v26; // rax
  unsigned int v27; // r12d
  __int64 v28; // rdx
  _BYTE *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v34; // [rsp+18h] [rbp-38h]

  v7 = *a2;
  if ( *a2 )
    goto LABEL_2;
  v9 = *a1;
  if ( *a1 != 46 )
    goto LABEL_11;
  v7 = *((_QWORD *)a1 - 8);
  if ( !v7 )
    goto LABEL_9;
  *a2 = v7;
  v23 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
  v25 = *v23;
  if ( (_BYTE)v25 == 17 )
  {
LABEL_31:
    v24 = (__int64)(v23 + 24);
LABEL_32:
    if ( *(_DWORD *)(a3 + 8) <= 0x40u && *(_DWORD *)(v24 + 8) <= 0x40u )
    {
      *(_QWORD *)a3 = *(_QWORD *)v24;
      *(_DWORD *)(a3 + 8) = *(_DWORD *)(v24 + 8);
    }
    else
    {
      sub_C43990(a3, v24);
    }
    return 1;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v23 + 1) + 8LL) - 17 > 1 || (unsigned __int8)v25 > 0x15u )
  {
LABEL_2:
    v8 = *a1;
  }
  else
  {
    v26 = sub_AD7630((__int64)v23, 0, v25);
    if ( v26 && *v26 == 17 )
      goto LABEL_42;
    v8 = *a1;
    v7 = *a2;
    v9 = *a1;
    if ( !*a2 )
      goto LABEL_11;
  }
  v9 = v8;
  if ( v8 != 46 )
  {
    v10 = *a2;
    goto LABEL_5;
  }
  v22 = *((_QWORD *)a1 - 8);
  if ( !v22 || v22 != v7 )
    goto LABEL_9;
  v23 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
  if ( *v23 == 17 )
    goto LABEL_31;
  v30 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v23 + 1) + 8LL) - 17;
  if ( (unsigned int)v30 > 1 || *v23 > 0x15u )
    goto LABEL_9;
  v26 = sub_AD7630((__int64)v23, 0, v30);
  if ( v26 && *v26 == 17 )
  {
LABEL_42:
    v24 = (__int64)(v26 + 24);
    goto LABEL_32;
  }
  v10 = *a2;
  v9 = *a1;
LABEL_5:
  if ( !v10 )
  {
LABEL_11:
    if ( v9 != 54 )
      goto LABEL_9;
    v10 = *((_QWORD *)a1 - 8);
    if ( !v10 )
      goto LABEL_9;
    *a2 = v10;
    v13 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
    v14 = *v13;
    if ( (_BYTE)v14 == 17 )
    {
LABEL_14:
      v15 = (__int64)(v13 + 24);
      goto LABEL_15;
    }
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v13 + 1) + 8LL) - 17 > 1 || (unsigned __int8)v14 > 0x15u )
      goto LABEL_59;
    v29 = sub_AD7630((__int64)v13, 0, v14);
    if ( v29 && *v29 == 17 )
      goto LABEL_53;
    v10 = *a2;
    if ( *a2 )
    {
LABEL_59:
      v9 = *a1;
      goto LABEL_6;
    }
LABEL_9:
    *a2 = 0;
    return 0;
  }
LABEL_6:
  if ( v9 != 54 )
    goto LABEL_9;
  v11 = *((_QWORD *)a1 - 8);
  if ( v11 != v10 || !v11 )
    goto LABEL_9;
  v13 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
  if ( *v13 == 17 )
    goto LABEL_14;
  v28 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v13 + 1) + 8LL) - 17;
  if ( (unsigned int)v28 > 1 )
    goto LABEL_9;
  if ( *v13 > 0x15u )
    goto LABEL_9;
  v29 = sub_AD7630((__int64)v13, 0, v28);
  if ( !v29 || *v29 != 17 )
    goto LABEL_9;
LABEL_53:
  v15 = (__int64)(v29 + 24);
LABEL_15:
  v16 = *(_DWORD *)(v15 + 8);
  v32 = v16;
  if ( v16 > 0x40 )
  {
    sub_C43690((__int64)&v31, 1, 0);
    v34 = v32;
    if ( v32 > 0x40 )
    {
      sub_C43780((__int64)&v33, (const void **)&v31);
      goto LABEL_18;
    }
  }
  else
  {
    v31 = 1;
    v34 = v16;
  }
  v33 = v31;
LABEL_18:
  sub_C47AC0((__int64)&v33, v15);
  if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
    j_j___libc_free_0_0(*(_QWORD *)a3);
  v17 = v32 <= 0x40;
  *(_QWORD *)a3 = v33;
  *(_DWORD *)(a3 + 8) = v34;
  if ( !v17 )
  {
    if ( v31 )
      j_j___libc_free_0_0(v31);
  }
  v18 = *(_DWORD *)(v15 + 8);
  v19 = v18 - 1;
  if ( v18 > 0x40 )
  {
    v27 = v18 - sub_C444A0(v15);
    v21 = 0;
    if ( v27 > 0x40 )
      goto LABEL_27;
    v20 = **(_QWORD ***)v15;
  }
  else
  {
    v20 = *(_QWORD **)v15;
  }
  v21 = v19 > (unsigned __int64)v20;
LABEL_27:
  *a4 = v21;
  return 1;
}
