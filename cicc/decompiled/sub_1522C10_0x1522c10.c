// Function: sub_1522C10
// Address: 0x1522c10
//
void __fastcall sub_1522C10(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 *v3; // rax
  __int64 v4; // r13
  __int64 v5; // r12
  _QWORD *v6; // r14
  __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // r15
  char v11; // al
  __int64 v12; // r9
  _QWORD *v13; // r8
  _QWORD *v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // [rsp+10h] [rbp-250h]
  _QWORD *v24; // [rsp+18h] [rbp-248h]
  _BYTE *v25; // [rsp+20h] [rbp-240h] BYREF
  __int64 v26; // [rsp+28h] [rbp-238h]
  _BYTE v27[560]; // [rsp+30h] [rbp-230h] BYREF

  v25 = v27;
  v26 = 0x4000000000LL;
  v1 = a1[4];
  if ( v1 != a1[3] )
  {
    while ( 1 )
    {
      v2 = *(unsigned int *)(v1 - 8);
      v3 = (__int64 *)(v1 - 16);
      v4 = *v3;
      v5 = *(_QWORD *)(*a1 + 24 * v2 + 16);
      a1[4] = v3;
      v6 = *(_QWORD **)(v4 + 8);
      if ( v6 )
        break;
LABEL_28:
      sub_164D160(v4, v5);
      sub_164BEC0(v4, v5, v18, v19, v20);
      v1 = a1[4];
      if ( v1 == a1[3] )
      {
        if ( v25 != v27 )
          _libc_free((unsigned __int64)v25);
        return;
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = (_QWORD *)sub_1648700(v6);
        v11 = *((_BYTE *)v10 + 16);
        if ( (unsigned __int8)(v11 - 4) <= 0xCu )
          break;
        if ( *v6 )
        {
          v7 = v6[1];
          v8 = v6[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v8 = v7;
          if ( v7 )
            *(_QWORD *)(v7 + 16) = *(_QWORD *)(v7 + 16) & 3LL | v8;
        }
        *v6 = v5;
        if ( v5 )
        {
          v9 = *(_QWORD *)(v5 + 8);
          v6[1] = v9;
          if ( v9 )
            *(_QWORD *)(v9 + 16) = (unsigned __int64)(v6 + 1) | *(_QWORD *)(v9 + 16) & 3LL;
          v6[2] = (v5 + 8) | v6[2] & 3LL;
          *(_QWORD *)(v5 + 8) = v6;
        }
        v6 = *(_QWORD **)(v4 + 8);
        if ( !v6 )
          goto LABEL_28;
      }
      v12 = 24LL * (*((_DWORD *)v10 + 5) & 0xFFFFFFF);
      if ( (*((_BYTE *)v10 + 23) & 0x40) != 0 )
      {
        v13 = (_QWORD *)*(v10 - 1);
        v14 = &v13[(unsigned __int64)v12 / 8];
      }
      else
      {
        v14 = v10;
        v13 = &v10[v12 / 0xFFFFFFFFFFFFFFF8LL];
      }
      v15 = (unsigned int)v26;
      if ( v13 != v14 )
        break;
LABEL_25:
      switch ( v11 )
      {
        case 6:
          v17 = sub_159DFD0(*v10, v25, v15);
          break;
        case 7:
          v17 = sub_159F090(*v10, v25, v15);
          break;
        case 8:
          v17 = sub_15A01B0(v25, v15);
          break;
        default:
          v17 = sub_15A47B0(v10, v25, v15, *v10, 0, 0);
          break;
      }
      sub_164D160(v10, v17);
      sub_159D850(v10);
      LODWORD(v26) = 0;
      v6 = *(_QWORD **)(v4 + 8);
      if ( !v6 )
        goto LABEL_28;
    }
    while ( 1 )
    {
      v16 = *v13;
      if ( *(_BYTE *)(*v13 + 16LL) != 5 || *(_WORD *)(v16 + 18) != 56 )
        goto LABEL_17;
      if ( v4 != v16 )
        break;
      v16 = v5;
      if ( HIDWORD(v26) <= (unsigned int)v15 )
      {
LABEL_23:
        v23 = v16;
        v24 = v13;
        sub_16CD150(&v25, v27, 0, 8);
        v15 = (unsigned int)v26;
        v16 = v23;
        v13 = v24;
      }
LABEL_18:
      v13 += 3;
      *(_QWORD *)&v25[8 * v15] = v16;
      v15 = (unsigned int)(v26 + 1);
      LODWORD(v26) = v26 + 1;
      if ( v13 == v14 )
      {
        v11 = *((_BYTE *)v10 + 16);
        goto LABEL_25;
      }
    }
    v21 = a1[3];
    v22 = a1[4];
    if ( v22 == v21 )
    {
LABEL_40:
      v16 = 0;
    }
    else
    {
      while ( *(_QWORD *)v21 != *v13 )
      {
        v21 += 16;
        if ( v22 == v21 )
          goto LABEL_40;
      }
      v16 = *(_QWORD *)(*a1 + 24LL * *(unsigned int *)(v21 + 8) + 16);
    }
LABEL_17:
    if ( HIDWORD(v26) <= (unsigned int)v15 )
      goto LABEL_23;
    goto LABEL_18;
  }
}
