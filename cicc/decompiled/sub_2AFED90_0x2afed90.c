// Function: sub_2AFED90
// Address: 0x2afed90
//
__int64 __fastcall sub_2AFED90(__int64 a1, char *a2, __int64 a3)
{
  unsigned __int8 v4; // cl
  char v5; // al
  unsigned int v6; // r15d
  __int64 v8; // r14
  unsigned __int8 *v9; // r13
  int v10; // r8d
  unsigned __int8 *v11; // rax
  __int64 v12; // rax
  bool v13; // di
  __int64 v14; // r9
  __int64 v15; // r15
  int v16; // edx
  char v17; // cl
  unsigned __int64 v18; // rbx
  char v19; // dl
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rsi
  unsigned __int64 v25; // rbx
  char v26; // r15
  __int64 v27; // rdx
  unsigned int v28; // ebx
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  char v35; // [rsp+3h] [rbp-5Dh]
  unsigned int v36; // [rsp+4h] [rbp-5Ch]
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h] BYREF
  __int64 v41; // [rsp+28h] [rbp-38h]

  v4 = *a2;
  if ( (unsigned __int8)*a2 > 0x1Cu )
  {
    v8 = 0;
    if ( (unsigned __int8)(v4 - 61) <= 1u )
      v8 = *((_QWORD *)a2 - 4);
    v5 = *(_BYTE *)a3;
    v9 = 0;
    if ( *(_BYTE *)a3 <= 0x1Cu )
      goto LABEL_11;
    if ( v5 == 61 )
    {
      v9 = *(unsigned __int8 **)(a3 - 32);
      goto LABEL_11;
    }
  }
  else
  {
    v5 = *(_BYTE *)a3;
    if ( *(_BYTE *)a3 <= 0x1Cu )
      return 0;
    v8 = 0;
    if ( v5 == 61 )
    {
      v9 = *(unsigned __int8 **)(a3 - 32);
      v8 = 0;
      v10 = -1;
      v11 = v9;
      goto LABEL_16;
    }
  }
  if ( v5 == 62 )
  {
    v9 = *(unsigned __int8 **)(a3 - 32);
    if ( v4 <= 0x1Cu )
    {
      v33 = *(_QWORD *)(a3 - 32);
      v10 = -1;
      goto LABEL_59;
    }
  }
  else
  {
    v9 = 0;
    if ( v4 <= 0x1Cu )
      return 0;
  }
LABEL_11:
  if ( v4 == 61 || v4 == 62 )
  {
    v32 = *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL);
    if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17 <= 1 )
      v32 = **(_QWORD **)(v32 + 16);
    v10 = *(_DWORD *)(v32 + 8) >> 8;
    if ( (unsigned __int8)v5 <= 0x1Cu )
      return 0;
    if ( v5 == 61 )
    {
LABEL_15:
      v11 = *(unsigned __int8 **)(a3 - 32);
LABEL_16:
      v12 = *((_QWORD *)v11 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
        v12 = **(_QWORD **)(v12 + 16);
      v13 = *(_DWORD *)(v12 + 8) >> 8 != v10 || v9 == 0 || v8 == 0;
      v5 = 61;
      goto LABEL_21;
    }
  }
  else
  {
    v10 = -1;
    if ( (unsigned __int8)v5 <= 0x1Cu )
    {
      if ( !v8 || !v9 )
        return 0;
      goto LABEL_23;
    }
    if ( v5 == 61 )
      goto LABEL_15;
  }
  if ( v5 != 62 )
  {
    v13 = v10 != -1 || v8 == 0 || v9 == 0;
    goto LABEL_21;
  }
  v33 = *(_QWORD *)(a3 - 32);
LABEL_59:
  v34 = *(_QWORD *)(v33 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 <= 1 )
    v34 = **(_QWORD **)(v34 + 16);
  v13 = *(_DWORD *)(v34 + 8) >> 8 != v10 || v9 == 0 || v8 == 0;
  v5 = 62;
LABEL_21:
  v6 = 0;
  if ( v13 )
    return v6;
  if ( v4 != 61 )
  {
LABEL_23:
    v14 = *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL);
    goto LABEL_24;
  }
  v14 = *((_QWORD *)a2 + 1);
LABEL_24:
  if ( v5 == 61 )
    v15 = *(_QWORD *)(a3 + 8);
  else
    v15 = *(_QWORD *)(*(_QWORD *)(a3 - 64) + 8LL);
  if ( v9 == (unsigned __int8 *)v8 )
    return 0;
  v16 = *(unsigned __int8 *)(v15 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 > 1 )
  {
    v17 = 0;
    if ( v16 == 18 )
      return 0;
  }
  else
  {
    v17 = 1;
    if ( v16 == 18 )
      goto LABEL_30;
  }
  if ( (v16 == 17) != v17 )
    return 0;
LABEL_30:
  v36 = v10;
  v37 = v14;
  v18 = (unsigned __int64)(sub_9208B0(*(_QWORD *)(a1 + 48), v15) + 7) >> 3;
  v35 = v19;
  v40 = sub_9208B0(*(_QWORD *)(a1 + 48), v37);
  v41 = v20;
  if ( (unsigned __int64)(v40 + 7) >> 3 != v18 || v35 != (_BYTE)v20 )
    return 0;
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
    v15 = **(_QWORD **)(v15 + 16);
  v21 = sub_9208B0(*(_QWORD *)(a1 + 48), v15);
  v22 = *(_QWORD *)(a1 + 48);
  v40 = v21;
  v41 = v23;
  v24 = v37;
  v25 = (unsigned __int64)(v21 + 7) >> 3;
  v26 = v23;
  if ( (unsigned int)*(unsigned __int8 *)(v37 + 8) - 17 <= 1 )
    v24 = **(_QWORD **)(v37 + 16);
  v40 = sub_9208B0(v22, v24);
  v41 = v27;
  if ( (unsigned __int64)(v40 + 7) >> 3 != v25 || v26 != (_BYTE)v27 )
    return 0;
  v28 = sub_AE2980(*(_QWORD *)(a1 + 48), v36)[3];
  v40 = sub_9C6480(*(_QWORD *)(a1 + 48), v37);
  v41 = v29;
  v30 = sub_CA1930(&v40);
  v39 = v28;
  if ( v28 > 0x40 )
    sub_C43690((__int64)&v38, v30, 0);
  else
    v38 = v30;
  LODWORD(v41) = v39;
  if ( v39 > 0x40 )
    sub_C43780((__int64)&v40, (const void **)&v38);
  else
    v40 = v38;
  LOBYTE(v31) = sub_2AFD350(a1, v8, v9, (__int64)&v40, 0);
  v6 = v31;
  if ( (unsigned int)v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  return v6;
}
