// Function: sub_3207D60
// Address: 0x3207d60
//
__int64 __fastcall sub_3207D60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v5; // al
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  __int64 v8; // r8
  __int64 v9; // rax
  unsigned __int8 **v10; // r14
  unsigned __int8 **v11; // r15
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // rdx
  unsigned __int64 v15; // r9
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r15
  __int64 v19; // rax
  int v20; // edi
  unsigned int *v21; // r14
  _DWORD *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // r15
  __int64 v25; // r14
  __int64 v26; // rax
  int v27; // eax
  char v28; // r9
  int v29; // ebx
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // r12d
  unsigned __int64 v34; // [rsp+0h] [rbp-B0h]
  __int64 v35; // [rsp+8h] [rbp-A8h]
  char v36; // [rsp+8h] [rbp-A8h]
  __int16 v37; // [rsp+14h] [rbp-9Ch]
  int v38; // [rsp+14h] [rbp-9Ch]
  int v39; // [rsp+18h] [rbp-98h]
  __int16 v40; // [rsp+22h] [rbp-8Eh] BYREF
  int v41; // [rsp+24h] [rbp-8Ch]
  char v42; // [rsp+28h] [rbp-88h]
  char v43; // [rsp+29h] [rbp-87h]
  __int16 v44; // [rsp+2Ah] [rbp-86h]
  int v45; // [rsp+2Ch] [rbp-84h]
  __int64 v46; // [rsp+30h] [rbp-80h] BYREF
  _DWORD *v47; // [rsp+38h] [rbp-78h]
  __int64 v48; // [rsp+40h] [rbp-70h]
  _DWORD *v49; // [rsp+48h] [rbp-68h]
  _BYTE *v50; // [rsp+50h] [rbp-60h] BYREF
  __int64 v51; // [rsp+58h] [rbp-58h]
  _BYTE v52[80]; // [rsp+60h] [rbp-50h] BYREF

  v51 = 0x800000000LL;
  v5 = *(_BYTE *)(a2 - 16);
  v50 = v52;
  if ( (v5 & 2) != 0 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 24LL);
    if ( v6 )
      goto LABEL_3;
LABEL_28:
    LODWORD(v46) = 3;
    v39 = 3;
LABEL_29:
    v17 = 0;
    LOWORD(v46) = 4609;
    v37 = 0;
    goto LABEL_30;
  }
  v6 = *(_QWORD *)(a2 - 16 - 8LL * ((v5 >> 2) & 0xF) + 24);
  if ( !v6 )
    goto LABEL_28;
LABEL_3:
  v7 = *(_BYTE *)(v6 - 16);
  if ( (v7 & 2) != 0 )
  {
    v8 = *(_QWORD *)(v6 - 32);
    v9 = *(unsigned int *)(v6 - 24);
  }
  else
  {
    v8 = v6 - 16 - 8LL * ((v7 >> 2) & 0xF);
    v9 = (*(_WORD *)(v6 - 16) >> 6) & 0xF;
  }
  v10 = (unsigned __int8 **)(v8 + 8 * v9);
  if ( (unsigned __int8 **)v8 == v10 )
    goto LABEL_28;
  v11 = (unsigned __int8 **)v8;
  do
  {
    v12 = sub_3206530(a1, *v11, 0);
    v14 = (unsigned int)v51;
    v15 = (unsigned int)v51 + 1LL;
    if ( v15 > HIDWORD(v51) )
    {
      v38 = v12;
      sub_C8D5F0((__int64)&v50, v52, (unsigned int)v51 + 1LL, 4u, v13, v15);
      v14 = (unsigned int)v51;
      v12 = v38;
    }
    a4 = (__int64)v50;
    ++v11;
    *(_DWORD *)&v50[4 * v14] = v12;
    v16 = (unsigned int)(v51 + 1);
    LODWORD(v51) = v51 + 1;
  }
  while ( v10 != v11 );
  v17 = (unsigned int)v16;
  if ( (unsigned int)v16 > 1 )
  {
    a4 = (__int64)v50;
    LODWORD(v46) = 3;
    v17 = (unsigned __int64)&v50[4 * (unsigned int)v16 - 4];
    if ( *(_DWORD *)v17 == 3 )
    {
      *(_DWORD *)v17 = 0;
      v16 = (unsigned int)v51;
    }
  }
  LODWORD(v46) = 3;
  v39 = 3;
  if ( !(_DWORD)v16 )
    goto LABEL_29;
  v18 = (unsigned __int64)v50;
  v19 = v16 - 1;
  a4 = 4 * v19;
  v37 = v19;
  v20 = *(_DWORD *)v50;
  v21 = (unsigned int *)(v50 + 4);
  v46 = 4609;
  v47 = 0;
  v39 = v20;
  v34 = (unsigned __int64)&v50[4 * v19 + 4];
  v48 = 0;
  v49 = 0;
  if ( !(4 * v19) )
  {
LABEL_30:
    v47 = 0;
    v23 = 0;
    v49 = 0;
    goto LABEL_20;
  }
  v35 = v19;
  v22 = (_DWORD *)sub_22077B0(4 * v19);
  v47 = v22;
  v23 = (__int64)v22;
  a4 = (__int64)&v22[v35];
  v49 = &v22[v35];
  if ( v21 != (unsigned int *)v34 )
  {
    do
    {
      if ( v22 )
      {
        a4 = *v21;
        *v22 = a4;
      }
      ++v21;
      ++v22;
    }
    while ( v21 != (unsigned int *)v34 );
    v23 = (__int64)v21 + v23 - v18 - 4;
  }
LABEL_20:
  v24 = a1 + 648;
  v48 = v23;
  v25 = a1 + 632;
  v26 = sub_37099F0(a1 + 648, &v46, v17, a4);
  v27 = sub_3707F80(a1 + 632, v26);
  v28 = 0;
  v29 = v27;
  v30 = (unsigned int)*(unsigned __int8 *)(a2 + 44) - 177;
  if ( (unsigned int)v30 <= 0xF )
    v28 = byte_44D4F40[v30];
  v36 = v28;
  v43 = sub_31F74D0(a2, 0, byte_3F871B3, 0);
  v41 = v39;
  v40 = 4104;
  v42 = v36;
  v44 = v37;
  v45 = v29;
  v31 = sub_37094D0(v24, &v40);
  v32 = sub_3707F80(v25, v31);
  if ( v47 )
    j_j___libc_free_0((unsigned __int64)v47);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
  return v32;
}
