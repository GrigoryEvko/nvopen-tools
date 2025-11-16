// Function: sub_2C75AD0
// Address: 0x2c75ad0
//
__int64 __fastcall sub_2C75AD0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  _BYTE *v6; // rax
  unsigned __int8 v7; // dl
  _BYTE **v8; // rax
  const char *v9; // rdx
  char *v10; // rsi
  size_t v11; // rdx
  char *v12; // rsi
  unsigned __int8 v14; // dl
  __int64 *v15; // rax
  __int64 v16; // rdx
  _QWORD **v17; // r15
  unsigned int v18; // eax
  __int64 v19; // rdi
  _BYTE *v20; // rax
  unsigned __int64 v21[2]; // [rsp+0h] [rbp-90h] BYREF
  _BYTE v22[16]; // [rsp+10h] [rbp-80h] BYREF
  char *v23; // [rsp+20h] [rbp-70h] BYREF
  size_t v24; // [rsp+28h] [rbp-68h]
  __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v26; // [rsp+38h] [rbp-58h]
  _BYTE *v27; // [rsp+40h] [rbp-50h]
  __int64 v28; // [rsp+48h] [rbp-48h]
  unsigned __int64 *v29; // [rsp+50h] [rbp-40h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v4 = sub_B10CD0(a2);
  if ( !v4 )
    return a1;
  v5 = *(_BYTE *)(v4 - 16);
  if ( (v5 & 2) != 0 )
  {
    v6 = **(_BYTE ***)(v4 - 32);
    if ( *v6 == 16 )
      goto LABEL_14;
  }
  else
  {
    v6 = *(_BYTE **)(v4 - 16 - 8LL * ((v5 >> 2) & 0xF));
    if ( *v6 == 16 )
      goto LABEL_14;
  }
  v7 = *(v6 - 16);
  if ( (v7 & 2) != 0 )
    v8 = (_BYTE **)*((_QWORD *)v6 - 4);
  else
    v8 = (_BYTE **)&v6[-8 * ((v7 >> 2) & 0xF) - 16];
  v6 = *v8;
  if ( !v6 )
  {
    v9 = byte_3F871B3;
    v10 = (char *)byte_3F871B3;
    goto LABEL_8;
  }
LABEL_14:
  v14 = *(v6 - 16);
  if ( (v14 & 2) != 0 )
    v15 = (__int64 *)*((_QWORD *)v6 - 4);
  else
    v15 = (__int64 *)&v6[-8 * ((v14 >> 2) & 0xF) - 16];
  if ( !*v15 || (v10 = (char *)sub_B91420(*v15), v9 = &v10[v16], !v10) )
  {
    LOBYTE(v25) = 0;
    v11 = 0;
    v23 = (char *)&v25;
    v12 = (char *)&v25;
    v24 = 0;
    goto LABEL_19;
  }
LABEL_8:
  v23 = (char *)&v25;
  sub_2C75590((__int64 *)&v23, v10, (__int64)v9);
  v11 = v24;
  v12 = v23;
LABEL_19:
  sub_2241490((unsigned __int64 *)a1, v12, v11);
  if ( v23 != (char *)&v25 )
    j_j___libc_free_0((unsigned __int64)v23);
  v29 = v21;
  v28 = 0x100000000LL;
  v21[0] = (unsigned __int64)v22;
  v21[1] = 0;
  v23 = (char *)&unk_49DD210;
  v22[0] = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  sub_CB5980((__int64)&v23, 0, 0, 0);
  if ( v26 == v27 )
  {
    v17 = (_QWORD **)sub_CB6200((__int64)&v23, (unsigned __int8 *)"(", 1u);
  }
  else
  {
    *v27 = 40;
    v17 = (_QWORD **)&v23;
    ++v27;
  }
  v18 = sub_B10CE0(a2);
  v19 = sub_CB59D0((__int64)v17, v18);
  v20 = *(_BYTE **)(v19 + 32);
  if ( *(_BYTE **)(v19 + 24) == v20 )
  {
    sub_CB6200(v19, (unsigned __int8 *)")", 1u);
  }
  else
  {
    *v20 = 41;
    ++*(_QWORD *)(v19 + 32);
  }
  sub_2241490((unsigned __int64 *)a1, (char *)*v29, v29[1]);
  v23 = (char *)&unk_49DD210;
  sub_CB5840((__int64)&v23);
  if ( (_BYTE *)v21[0] != v22 )
    j_j___libc_free_0(v21[0]);
  return a1;
}
