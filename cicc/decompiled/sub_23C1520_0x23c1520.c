// Function: sub_23C1520
// Address: 0x23c1520
//
__int64 *__fastcall sub_23C1520(__int64 *a1, __int64 a2, __int64 a3)
{
  char v3; // bl
  __int64 v4; // r12
  unsigned int v5; // r13d
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 *v18; // r14
  __int64 v19; // rax
  _QWORD *v20; // rdi
  __int64 *v21; // rbx
  __int64 v22; // rsi
  __int64 *v23; // r14
  __int64 v24; // rax
  _QWORD *v25; // rdi
  unsigned int v26; // eax
  int v27; // [rsp+0h] [rbp-A0h]
  unsigned int v28; // [rsp+4h] [rbp-9Ch]
  __int64 v29; // [rsp+8h] [rbp-98h]
  int v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+10h] [rbp-90h]
  __int64 v33; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v34; // [rsp+28h] [rbp-78h]
  __int64 v35; // [rsp+30h] [rbp-70h]
  unsigned int v36; // [rsp+38h] [rbp-68h]
  char v37; // [rsp+40h] [rbp-60h]
  __int64 v38; // [rsp+48h] [rbp-58h]
  __int64 v39; // [rsp+50h] [rbp-50h]
  __int64 v40; // [rsp+58h] [rbp-48h]
  unsigned int v41; // [rsp+60h] [rbp-40h]

  sub_23C0920((__int64)&v33, a3, 1);
  v3 = v37;
  if ( v37 )
  {
    v15 = (__int64)v34;
    ++v33;
    v34 = 0;
    v29 = v15;
    LODWORD(v15) = HIDWORD(v35);
    v27 = v35;
    v35 = 0;
    v30 = v15;
    LODWORD(v15) = v36;
    v36 = 0;
    v28 = v15;
  }
  v4 = v39;
  v5 = v41;
  v39 = 0;
  v6 = v40;
  ++v38;
  v40 = 0;
  v41 = 0;
  v7 = sub_22077B0(0x50u);
  v8 = v7;
  if ( v7 )
  {
    *(_BYTE *)(v7 + 40) = 0;
    *(_QWORD *)v7 = &unk_4A161A0;
    if ( v3 )
    {
      *(_QWORD *)(v7 + 8) = 1;
      *(_BYTE *)(v7 + 40) = 1;
      *(_QWORD *)(v7 + 16) = v29;
      v29 = 0;
      *(_DWORD *)(v7 + 24) = v27;
      *(_DWORD *)(v7 + 28) = v30;
      v26 = v28;
      v28 = 0;
      *(_DWORD *)(v8 + 32) = v26;
    }
    *(_QWORD *)(v8 + 48) = 1;
    *(_QWORD *)(v8 + 64) = v6;
    *(_DWORD *)(v8 + 72) = v5;
    v31 = 0;
    *(_QWORD *)(v8 + 56) = v4;
    v4 = 0;
  }
  else
  {
    v31 = 40LL * v5;
    if ( v5 )
    {
      v13 = v4 + 40LL * v5;
      v14 = v4;
      do
      {
        if ( *(_QWORD *)v14 != -8192 && *(_QWORD *)v14 != -4096 )
          sub_C7D6A0(*(_QWORD *)(v14 + 16), 16LL * *(unsigned int *)(v14 + 32), 8);
        v14 += 40;
      }
      while ( v13 != v14 );
    }
  }
  sub_C7D6A0(v4, v31, 8);
  if ( v3 )
  {
    v21 = (__int64 *)v29;
    v22 = 40LL * v28;
    v23 = (__int64 *)(v29 + v22);
    if ( v28 )
    {
      do
      {
        while ( 1 )
        {
          if ( *v21 <= 0x7FFFFFFFFFFFFFFDLL )
          {
            v21[1] = (__int64)&unk_49DB368;
            v24 = v21[4];
            if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
              break;
          }
          v21 += 5;
          if ( v23 == v21 )
            goto LABEL_40;
        }
        v25 = v21 + 2;
        v21 += 5;
        sub_BD60C0(v25);
      }
      while ( v23 != v21 );
    }
LABEL_40:
    sub_C7D6A0(v29, v22, 8);
  }
  *a1 = v8;
  v9 = v41;
  if ( v41 )
  {
    v10 = v39;
    v11 = v39 + 40LL * v41;
    do
    {
      if ( *(_QWORD *)v10 != -4096 && *(_QWORD *)v10 != -8192 )
        sub_C7D6A0(*(_QWORD *)(v10 + 16), 16LL * *(unsigned int *)(v10 + 32), 8);
      v10 += 40;
    }
    while ( v11 != v10 );
    v9 = v41;
  }
  sub_C7D6A0(v39, 40 * v9, 8);
  if ( v37 )
  {
    v16 = v36;
    v37 = 0;
    if ( v36 )
    {
      v17 = v34;
      v18 = &v34[5 * v36];
      do
      {
        while ( 1 )
        {
          if ( *v17 <= 0x7FFFFFFFFFFFFFFDLL )
          {
            v17[1] = (__int64)&unk_49DB368;
            v19 = v17[4];
            if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
              break;
          }
          v17 += 5;
          if ( v18 == v17 )
            goto LABEL_31;
        }
        v20 = v17 + 2;
        v17 += 5;
        sub_BD60C0(v20);
      }
      while ( v18 != v17 );
LABEL_31:
      v16 = v36;
    }
    sub_C7D6A0((__int64)v34, 40 * v16, 8);
  }
  return a1;
}
