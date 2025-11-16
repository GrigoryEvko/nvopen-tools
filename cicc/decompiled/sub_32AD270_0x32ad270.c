// Function: sub_32AD270
// Address: 0x32ad270
//
__int64 __fastcall sub_32AD270(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rcx
  int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r15
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r12
  bool v14; // zf
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // r9d
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int16 v21; // [rsp+16h] [rbp-AAh]
  __int64 v22; // [rsp+18h] [rbp-A8h]
  __int64 v23; // [rsp+20h] [rbp-A0h] BYREF
  int v24; // [rsp+28h] [rbp-98h]
  __int128 v25; // [rsp+30h] [rbp-90h] BYREF
  __int128 v26; // [rsp+40h] [rbp-80h] BYREF
  __int64 v27; // [rsp+50h] [rbp-70h] BYREF
  int v28; // [rsp+58h] [rbp-68h]
  int v29; // [rsp+60h] [rbp-60h]
  __int128 *v30; // [rsp+68h] [rbp-58h]
  char v31; // [rsp+74h] [rbp-4Ch]
  __int128 *v32; // [rsp+78h] [rbp-48h]
  char v33; // [rsp+84h] [rbp-3Ch]
  char v34; // [rsp+8Ch] [rbp-34h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *v4;
  v6 = *((_DWORD *)v4 + 2);
  v7 = *(_QWORD *)(a2 + 48);
  v8 = *(_QWORD *)(a2 + 80);
  v22 = v5;
  LOWORD(v5) = *(_WORD *)v7;
  v9 = *(_QWORD *)(v7 + 8);
  v23 = v8;
  v21 = v5;
  if ( v8 )
    sub_B96E90((__int64)&v23, v8, 1);
  v10 = *a1;
  v24 = *(_DWORD *)(a2 + 72);
  v28 = v6;
  v27 = v22;
  v11 = sub_3402EA0(v10, 201, (unsigned int)&v23, v21, v9, 0, (__int64)&v27, 1);
  if ( v11 )
  {
    v12 = v11;
    goto LABEL_5;
  }
  if ( *(_DWORD *)(v22 + 24) == 201 )
  {
    v12 = **(_QWORD **)(v22 + 40);
    goto LABEL_5;
  }
  v14 = *((_BYTE *)a1 + 33) == 0;
  *(_QWORD *)&v26 = 0;
  *(_QWORD *)&v25 = 0;
  DWORD2(v25) = 0;
  DWORD2(v26) = 0;
  if ( !v14 )
  {
    v15 = a1[1];
    v16 = 1;
    if ( v21 != 1 )
    {
      if ( !v21 )
      {
LABEL_17:
        v12 = 0;
        goto LABEL_5;
      }
      v16 = v21;
      if ( !*(_QWORD *)(v15 + 8LL * v21 + 112) )
        goto LABEL_20;
    }
    if ( *(_BYTE *)(v15 + 500 * v16 + 6604) )
      goto LABEL_20;
  }
  v32 = &v26;
  v30 = &v25;
  LODWORD(v27) = 201;
  v28 = 192;
  v29 = 201;
  v31 = 0;
  v33 = 0;
  v34 = 0;
  if ( !sub_32AD1B0(a2, 0, (__int64)&v27) )
  {
    if ( !*((_BYTE *)a1 + 33) )
    {
LABEL_16:
      v30 = &v25;
      LODWORD(v27) = 201;
      v28 = 190;
      v29 = 201;
      v31 = 0;
      v32 = &v26;
      v33 = 0;
      v34 = 0;
      if ( !sub_32AD1B0(a2, 0, (__int64)&v27) )
        goto LABEL_17;
      v20 = sub_3406EB0(*a1, 192, (unsigned int)&v23, v21, v9, v18, v25, v26);
      goto LABEL_26;
    }
    v15 = a1[1];
LABEL_20:
    v19 = 1;
    if ( v21 != 1 )
    {
      if ( !v21 )
        goto LABEL_17;
      v19 = v21;
      if ( !*(_QWORD *)(v15 + 8LL * v21 + 112) )
        goto LABEL_17;
    }
    if ( *(_BYTE *)(v15 + 500 * v19 + 6606) )
      goto LABEL_17;
    goto LABEL_16;
  }
  v20 = sub_3406EB0(*a1, 190, (unsigned int)&v23, v21, v9, v17, v25, v26);
LABEL_26:
  v12 = v20;
LABEL_5:
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v12;
}
