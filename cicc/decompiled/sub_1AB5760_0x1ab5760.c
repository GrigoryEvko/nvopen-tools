// Function: sub_1AB5760
// Address: 0x1ab5760
//
__int64 __fastcall sub_1AB5760(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  __int64 v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // r15
  __int64 v12; // rsi
  __int64 v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // rdx
  char v18; // cl
  char v19; // cl
  __int64 v20; // rbx
  __int64 v21; // rdx
  char v22; // al
  __int64 *v23; // rsi
  char v24; // al
  __int64 v25; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  char v29; // al
  __int64 *v30; // rcx
  __int64 v35; // [rsp+30h] [rbp-80h]
  __int64 v37; // [rsp+40h] [rbp-70h]
  char v38; // [rsp+4Ch] [rbp-64h]
  char v39; // [rsp+4Eh] [rbp-62h]
  char v40; // [rsp+4Fh] [rbp-61h]
  const char *v41; // [rsp+50h] [rbp-60h] BYREF
  __int64 v42; // [rsp+58h] [rbp-58h]
  const char **v43; // [rsp+60h] [rbp-50h] BYREF
  __int64 *v44; // [rsp+68h] [rbp-48h]
  __int16 v45; // [rsp+70h] [rbp-40h]

  v35 = a4;
  v45 = 257;
  v7 = sub_157E9C0(a1);
  v8 = (_QWORD *)sub_22077B0(64);
  v9 = (__int64)v8;
  if ( v8 )
    sub_157FB60(v8, v7, (__int64)&v43, a4, 0);
  if ( (*(_BYTE *)(a1 + 23) & 0x20) != 0 )
  {
    v41 = sub_1649960(a1);
    v42 = v28;
    v29 = *((_BYTE *)a3 + 16);
    if ( v29 )
    {
      if ( v29 == 1 )
      {
        v43 = &v41;
        v45 = 261;
      }
      else
      {
        if ( *((_BYTE *)a3 + 17) == 1 )
        {
          v30 = (__int64 *)*a3;
        }
        else
        {
          v30 = a3;
          v29 = 2;
        }
        v44 = v30;
        v43 = &v41;
        LOBYTE(v45) = 5;
        HIBYTE(v45) = v29;
      }
    }
    else
    {
      v45 = 256;
    }
    sub_164B780(v9, (__int64 *)&v43);
  }
  if ( v35 )
    v35 = *(_QWORD *)(v35 + 40);
  v10 = *(_QWORD *)(a1 + 48);
  v37 = a1 + 40;
  if ( v10 == a1 + 40 )
  {
    if ( !a5 )
      goto LABEL_40;
    v24 = a5[1];
    goto LABEL_39;
  }
  v40 = 0;
  v39 = 0;
  v38 = 0;
  do
  {
    v20 = v10 - 24;
    if ( !v10 )
      v20 = 0;
    if ( v35 != 0 && a6 != 0 )
      sub_15ABE10(a6, v35, v20);
    v11 = sub_15F4880(v20);
    if ( (*(_BYTE *)(v20 + 23) & 0x20) != 0 )
    {
      v41 = sub_1649960(v20);
      v42 = v21;
      v22 = *((_BYTE *)a3 + 16);
      if ( v22 )
      {
        if ( v22 == 1 )
        {
          v43 = &v41;
          v45 = 261;
        }
        else
        {
          if ( *((_BYTE *)a3 + 17) == 1 )
          {
            v23 = (__int64 *)*a3;
          }
          else
          {
            v23 = a3;
            v22 = 2;
          }
          v44 = v23;
          v43 = &v41;
          LOBYTE(v45) = 5;
          HIBYTE(v45) = v22;
        }
      }
      else
      {
        v45 = 256;
      }
      sub_164B780(v11, (__int64 *)&v43);
    }
    sub_157E9D0(v9 + 40, v11);
    v12 = *(_QWORD *)(v9 + 40);
    v13 = *(_QWORD *)(v11 + 24);
    *(_QWORD *)(v11 + 32) = v9 + 40;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v11 + 24) = v12 | v13 & 7;
    *(_QWORD *)(v12 + 8) = v11 + 24;
    *(_QWORD *)(v9 + 40) = *(_QWORD *)(v9 + 40) & 7LL | (v11 + 24);
    v14 = sub_1AB4240(a2, v20);
    v15 = v14[2];
    if ( v11 != v15 )
    {
      if ( v15 != 0 && v15 != -8 && v15 != -16 )
        sub_1649B30(v14);
      v14[2] = v11;
      if ( v11 != -16 && v11 != -8 )
        sub_164C220((__int64)v14);
    }
    v16 = *(_BYTE *)(v20 + 16);
    if ( v16 == 78 )
    {
      v27 = *(_QWORD *)(v20 - 24);
      if ( *(_BYTE *)(v27 + 16) || (*(_BYTE *)(v27 + 33) & 0x20) == 0 )
        v38 = 1;
      else
        v38 |= (unsigned int)(*(_DWORD *)(v27 + 36) - 35) > 3;
    }
    else if ( v16 == 53 )
    {
      v17 = *(_QWORD *)(v20 - 24);
      v18 = v39;
      if ( *(_BYTE *)(v17 + 16) != 13 )
        v18 = 1;
      v39 = v18;
      v19 = v40;
      if ( *(_BYTE *)(v17 + 16) == 13 )
        v19 = 1;
      v40 = v19;
    }
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v37 != v10 );
  if ( a5 )
  {
    *a5 |= v38;
    v24 = a5[1] | v39;
    a5[1] = v24;
    if ( v40 )
    {
      v25 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 80LL);
      if ( v25 )
        v24 |= a1 != v25 - 24;
      else
        v24 = v40;
    }
LABEL_39:
    a5[1] = v24;
  }
LABEL_40:
  j___libc_free_0(0);
  return v9;
}
