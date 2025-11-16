// Function: sub_AF3990
// Address: 0xaf3990
//
__int64 __fastcall sub_AF3990(unsigned int a1, __int64 a2)
{
  unsigned int v2; // r12d
  int v3; // r13d
  int v4; // r13d
  int v5; // r13d
  int v6; // r13d
  int v7; // r13d
  int v8; // r13d
  int v9; // r13d
  int v10; // r13d
  int v11; // r13d
  int v12; // r13d
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax

  v2 = a1;
  if ( (a1 & 1) != 0 )
  {
    v14 = *(unsigned int *)(a2 + 8);
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, a2 + 16, v14 + 1, 4);
      v14 = *(unsigned int *)(a2 + 8);
    }
    v2 = a1 & 0xFFE;
    *(_DWORD *)(*(_QWORD *)a2 + 4 * v14) = 1;
    ++*(_DWORD *)(a2 + 8);
    v3 = a1 & 2;
    if ( (a1 & 2) == 0 )
    {
LABEL_3:
      v4 = v2 & 4;
      if ( (v2 & 4) == 0 )
        goto LABEL_4;
      goto LABEL_19;
    }
  }
  else
  {
    v3 = a1 & 2;
    if ( (a1 & 2) == 0 )
      goto LABEL_3;
  }
  v15 = *(unsigned int *)(a2 + 8);
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v15 + 1, 4);
    v15 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v15) = v3;
  v2 &= v3 ^ 0xFFF;
  ++*(_DWORD *)(a2 + 8);
  v4 = v2 & 4;
  if ( (v2 & 4) == 0 )
  {
LABEL_4:
    v5 = v2 & 8;
    if ( (v2 & 8) == 0 )
      goto LABEL_5;
    goto LABEL_22;
  }
LABEL_19:
  v16 = *(unsigned int *)(a2 + 8);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v16 + 1, 4);
    v16 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v16) = v4;
  v2 &= v4 ^ 0xFFF;
  ++*(_DWORD *)(a2 + 8);
  v5 = v2 & 8;
  if ( (v2 & 8) == 0 )
  {
LABEL_5:
    v6 = v2 & 0x10;
    if ( (v2 & 0x10) == 0 )
      goto LABEL_6;
    goto LABEL_25;
  }
LABEL_22:
  v17 = *(unsigned int *)(a2 + 8);
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v17 + 1, 4);
    v17 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v17) = v5;
  v2 &= v5 ^ 0xFFF;
  ++*(_DWORD *)(a2 + 8);
  v6 = v2 & 0x10;
  if ( (v2 & 0x10) == 0 )
  {
LABEL_6:
    v7 = v2 & 0x20;
    if ( (v2 & 0x20) == 0 )
      goto LABEL_7;
    goto LABEL_28;
  }
LABEL_25:
  v18 = *(unsigned int *)(a2 + 8);
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v18 + 1, 4);
    v18 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v18) = v6;
  v2 &= v6 ^ 0xFFF;
  ++*(_DWORD *)(a2 + 8);
  v7 = v2 & 0x20;
  if ( (v2 & 0x20) == 0 )
  {
LABEL_7:
    v8 = v2 & 0x40;
    if ( (v2 & 0x40) == 0 )
      goto LABEL_8;
    goto LABEL_31;
  }
LABEL_28:
  v19 = *(unsigned int *)(a2 + 8);
  if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v19 + 1, 4);
    v19 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v19) = v7;
  v2 &= v7 ^ 0xFFF;
  ++*(_DWORD *)(a2 + 8);
  v8 = v2 & 0x40;
  if ( (v2 & 0x40) == 0 )
  {
LABEL_8:
    v9 = v2 & 0x80;
    if ( (v2 & 0x80) == 0 )
      goto LABEL_9;
    goto LABEL_34;
  }
LABEL_31:
  v20 = *(unsigned int *)(a2 + 8);
  if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v20 + 1, 4);
    v20 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v20) = v8;
  v2 &= v8 ^ 0xFFF;
  ++*(_DWORD *)(a2 + 8);
  v9 = v2 & 0x80;
  if ( (v2 & 0x80) == 0 )
  {
LABEL_9:
    v10 = v2 & 0x100;
    if ( (v2 & 0x100) == 0 )
      goto LABEL_10;
    goto LABEL_37;
  }
LABEL_34:
  v21 = *(unsigned int *)(a2 + 8);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v21 + 1, 4);
    v21 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v21) = v9;
  v2 &= v9 ^ 0xFFF;
  ++*(_DWORD *)(a2 + 8);
  v10 = v2 & 0x100;
  if ( (v2 & 0x100) == 0 )
  {
LABEL_10:
    v11 = v2 & 0x200;
    if ( (v2 & 0x200) == 0 )
      goto LABEL_11;
LABEL_40:
    v23 = *(unsigned int *)(a2 + 8);
    if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, a2 + 16, v23 + 1, 4);
      v23 = *(unsigned int *)(a2 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * v23) = v11;
    v2 &= v11 ^ 0xFFF;
    ++*(_DWORD *)(a2 + 8);
    v12 = v2 & 0x800;
    if ( (v2 & 0x800) == 0 )
      return v2;
    goto LABEL_43;
  }
LABEL_37:
  v22 = *(unsigned int *)(a2 + 8);
  if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v22 + 1, 4);
    v22 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v22) = v10;
  v2 &= v10 ^ 0xFFF;
  ++*(_DWORD *)(a2 + 8);
  v11 = v2 & 0x200;
  if ( (v2 & 0x200) != 0 )
    goto LABEL_40;
LABEL_11:
  v12 = v2 & 0x800;
  if ( (v2 & 0x800) == 0 )
    return v2;
LABEL_43:
  v24 = *(unsigned int *)(a2 + 8);
  if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v24 + 1, 4);
    v24 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v24) = v12;
  ++*(_DWORD *)(a2 + 8);
  return (v12 ^ 0xFFF) & v2;
}
