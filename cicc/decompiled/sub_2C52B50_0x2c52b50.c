// Function: sub_2C52B50
// Address: 0x2c52b50
//
__int64 __fastcall sub_2C52B50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r10
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rdi
  __int16 v21; // cx
  __int64 v22; // rsi
  __int64 v23; // r9
  __int64 v24; // r13
  unsigned __int64 v25; // rsi
  unsigned int *v26; // rax
  int v27; // ecx
  unsigned int *v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // r9
  __int64 v31; // r13
  unsigned __int64 v32; // rsi
  unsigned int *v33; // rax
  int v34; // ecx
  unsigned int *v35; // rdx
  __int64 result; // rax
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rsi
  unsigned __int64 v40; // rsi
  __int64 v41; // [rsp+8h] [rbp-E8h]
  _QWORD v42[4]; // [rsp+20h] [rbp-D0h] BYREF
  __int16 v43; // [rsp+40h] [rbp-B0h]
  _QWORD v44[4]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v45; // [rsp+70h] [rbp-80h]
  __int64 v46; // [rsp+80h] [rbp-70h]
  _QWORD v47[2]; // [rsp+88h] [rbp-68h] BYREF
  __int64 v48; // [rsp+98h] [rbp-58h]
  __int64 v49; // [rsp+A0h] [rbp-50h]
  __int16 v50; // [rsp+A8h] [rbp-48h]
  __int64 v51[8]; // [rsp+B0h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 48);
  v46 = a2;
  v47[0] = 0;
  v47[1] = 0;
  v48 = v4;
  if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
    sub_BD73F0((__int64)v47);
  v5 = *(_QWORD *)(a2 + 56);
  v50 = *(_WORD *)(a2 + 64);
  v49 = v5;
  sub_B33910(v51, (__int64 *)a2);
  sub_D5F1F0(a2, a3);
  v42[0] = sub_BD5D20(*(_QWORD *)(a1 + 8));
  v42[2] = ".frozen";
  v43 = 773;
  v6 = *(_QWORD *)(a1 + 8);
  v42[1] = v7;
  v41 = v6;
  v45 = 257;
  v8 = sub_BD2C40(72, 1u);
  v9 = (__int64)v8;
  if ( v8 )
    sub_B549F0((__int64)v8, v41, (__int64)v44, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v9,
    v42,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v10 = *(_QWORD *)a2;
  v11 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v11 )
  {
    do
    {
      v12 = *(_QWORD *)(v10 + 8);
      v13 = *(_DWORD *)v10;
      v10 += 16;
      sub_B99FD0(v9, v13, v12);
    }
    while ( v11 != v10 );
  }
  v14 = 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
  {
    v15 = *(_QWORD *)(a3 - 8);
    a3 = v15 + v14;
  }
  else
  {
    v15 = a3 - v14;
  }
  for ( ; a3 != v15; v15 += 32 )
  {
    while ( 1 )
    {
      if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)v15 )
      {
        if ( *(_QWORD *)v15 )
        {
          v16 = *(_QWORD *)(v15 + 8);
          **(_QWORD **)(v15 + 16) = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 16) = *(_QWORD *)(v15 + 16);
        }
        *(_QWORD *)v15 = v9;
        if ( v9 )
          break;
      }
      v15 += 32;
      if ( a3 == v15 )
        goto LABEL_21;
    }
    v17 = *(_QWORD *)(v9 + 16);
    *(_QWORD *)(v15 + 8) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = v15 + 8;
    *(_QWORD *)(v15 + 16) = v9 + 16;
    *(_QWORD *)(v9 + 16) = v15;
  }
LABEL_21:
  v18 = v48;
  v19 = v46;
  *(_QWORD *)(a1 + 8) = 0;
  v20 = v49;
  v21 = v50;
  if ( !v18 )
  {
    *(_QWORD *)(v19 + 48) = 0;
    v29 = v51[0];
    *(_QWORD *)(v19 + 56) = 0;
    *(_WORD *)(v19 + 64) = 0;
    v44[0] = v29;
    if ( !v29 )
      goto LABEL_50;
    goto LABEL_35;
  }
  *(_QWORD *)(v19 + 48) = v18;
  *(_QWORD *)(v19 + 56) = v20;
  *(_WORD *)(v19 + 64) = v21;
  if ( v20 != v18 + 48 )
  {
    if ( v20 )
      v20 -= 24;
    v22 = *(_QWORD *)sub_B46C60(v20);
    v44[0] = v22;
    if ( v22 && (sub_B96E90((__int64)v44, v22, 1), (v24 = v44[0]) != 0) )
    {
      v25 = *(unsigned int *)(v19 + 8);
      v26 = *(unsigned int **)v19;
      v27 = *(_DWORD *)(v19 + 8);
      v28 = (unsigned int *)(*(_QWORD *)v19 + 16 * v25);
      if ( *(unsigned int **)v19 != v28 )
      {
        while ( 1 )
        {
          v23 = *v26;
          if ( !(_DWORD)v23 )
            break;
          v26 += 4;
          if ( v28 == v26 )
            goto LABEL_60;
        }
        *((_QWORD *)v26 + 1) = v44[0];
LABEL_32:
        sub_B91220((__int64)v44, v24);
LABEL_33:
        v19 = v46;
        goto LABEL_34;
      }
LABEL_60:
      v38 = *(unsigned int *)(v19 + 12);
      if ( v25 >= v38 )
      {
        v40 = v25 + 1;
        if ( v38 < v40 )
        {
          sub_C8D5F0(v19, (const void *)(v19 + 16), v40, 0x10u, v19 + 16, v23);
          v28 = (unsigned int *)(*(_QWORD *)v19 + 16LL * *(unsigned int *)(v19 + 8));
        }
        *(_QWORD *)v28 = 0;
        *((_QWORD *)v28 + 1) = v24;
        ++*(_DWORD *)(v19 + 8);
        v24 = v44[0];
      }
      else
      {
        if ( v28 )
        {
          *v28 = 0;
          *((_QWORD *)v28 + 1) = v24;
          v27 = *(_DWORD *)(v19 + 8);
          v24 = v44[0];
        }
        *(_DWORD *)(v19 + 8) = v27 + 1;
      }
    }
    else
    {
      sub_93FB40(v19, 0);
      v24 = v44[0];
    }
    if ( !v24 )
      goto LABEL_33;
    goto LABEL_32;
  }
LABEL_34:
  v29 = v51[0];
  v44[0] = v51[0];
  if ( !v51[0] )
  {
LABEL_50:
    sub_93FB40(v19, 0);
    v31 = v44[0];
    goto LABEL_51;
  }
LABEL_35:
  sub_B96E90((__int64)v44, v29, 1);
  v31 = v44[0];
  if ( !v44[0] )
    goto LABEL_50;
  v32 = *(unsigned int *)(v19 + 8);
  v33 = *(unsigned int **)v19;
  v34 = *(_DWORD *)(v19 + 8);
  v35 = (unsigned int *)(*(_QWORD *)v19 + 16 * v32);
  if ( *(unsigned int **)v19 != v35 )
  {
    while ( *v33 )
    {
      v33 += 4;
      if ( v35 == v33 )
        goto LABEL_53;
    }
    *((_QWORD *)v33 + 1) = v44[0];
    goto LABEL_41;
  }
LABEL_53:
  v37 = *(unsigned int *)(v19 + 12);
  if ( v32 >= v37 )
  {
    v39 = v32 + 1;
    if ( v37 < v39 )
    {
      sub_C8D5F0(v19, (const void *)(v19 + 16), v39, 0x10u, v19 + 16, v30);
      v35 = (unsigned int *)(*(_QWORD *)v19 + 16LL * *(unsigned int *)(v19 + 8));
    }
    *(_QWORD *)v35 = 0;
    *((_QWORD *)v35 + 1) = v31;
    ++*(_DWORD *)(v19 + 8);
    v31 = v44[0];
  }
  else
  {
    if ( v35 )
    {
      *v35 = 0;
      *((_QWORD *)v35 + 1) = v31;
      v34 = *(_DWORD *)(v19 + 8);
      v31 = v44[0];
    }
    *(_DWORD *)(v19 + 8) = v34 + 1;
  }
LABEL_51:
  if ( v31 )
LABEL_41:
    sub_B91220((__int64)v44, v31);
  if ( v51[0] )
    sub_B91220((__int64)v51, v51[0]);
  result = v48;
  if ( v48 != 0 && v48 != -4096 && v48 != -8192 )
    return sub_BD60C0(v47);
  return result;
}
