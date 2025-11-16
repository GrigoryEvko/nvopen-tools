// Function: sub_1FD7E90
// Address: 0x1fd7e90
//
__int64 __fastcall sub_1FD7E90(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // al
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r10
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // rbx
  int v17; // eax
  __int64 v18; // r15
  __int64 *v19; // r14
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // r14
  int v24; // eax
  __int64 v25; // rax
  char v26; // di
  unsigned int v27; // eax
  int v28; // edx
  int v29; // r11d
  __int64 v30; // [rsp+8h] [rbp-128h]
  __int64 v31; // [rsp+18h] [rbp-118h]
  __int64 v33; // [rsp+38h] [rbp-F8h]
  int v34; // [rsp+40h] [rbp-F0h]
  unsigned int v35; // [rsp+44h] [rbp-ECh]
  __int64 v36; // [rsp+48h] [rbp-E8h]
  char v37; // [rsp+5Bh] [rbp-D5h] BYREF
  unsigned int v38; // [rsp+5Ch] [rbp-D4h] BYREF
  __int64 v39; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v40; // [rsp+68h] [rbp-C8h]
  _QWORD v41[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v42; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v43; // [rsp+88h] [rbp-A8h]
  _BYTE v44[8]; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+98h] [rbp-98h]
  __int64 v46; // [rsp+A0h] [rbp-90h]
  unsigned __int64 v47[2]; // [rsp+B0h] [rbp-80h] BYREF
  _BYTE v48[112]; // [rsp+C0h] [rbp-70h] BYREF

  v3 = sub_1FD35E0(*(_QWORD *)(a1 + 96), *(_QWORD *)a2);
  if ( !v3 || !*(_QWORD *)(*(_QWORD *)(a1 + 112) + 8LL * v3 + 120) && v3 != 2 )
    return 0;
  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(_QWORD *)(a2 - 24);
  v6 = *(unsigned int *)(v4 + 232);
  v7 = *(_QWORD *)v5;
  if ( !(_DWORD)v6 )
  {
LABEL_24:
    if ( *(_BYTE *)(v5 + 16) > 0x17u )
    {
      v35 = sub_1FD4520(v4, (__int64 *)v5);
      goto LABEL_8;
    }
    return 0;
  }
  v8 = *(_QWORD *)(v4 + 216);
  v9 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( v5 != *v10 )
  {
    v28 = 1;
    while ( v11 != -8 )
    {
      v29 = v28 + 1;
      v9 = (v6 - 1) & (v28 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v5 == *v10 )
        goto LABEL_6;
      v28 = v29;
    }
    goto LABEL_24;
  }
LABEL_6:
  if ( v10 == (__int64 *)(v8 + 16 * v6) )
    goto LABEL_24;
  v35 = *((_DWORD *)v10 + 2);
LABEL_8:
  v12 = sub_20C7BE0(v7, *(_QWORD *)(a2 + 56), *(_QWORD *)(a2 + 56) + 4LL * *(unsigned int *)(a2 + 64), 0);
  v13 = *(_QWORD *)(a1 + 96);
  v14 = v12;
  v15 = *(_QWORD *)(a1 + 112);
  v47[0] = (unsigned __int64)v48;
  v47[1] = 0x400000000LL;
  sub_20C7CE0(v15, v13, v7, v47, 0, 0);
  if ( (_DWORD)v14 )
  {
    v33 = 16 * v14;
    v16 = 0;
    do
    {
      v18 = *(_QWORD *)(a1 + 112);
      v19 = (__int64 *)(v16 + v47[0]);
      v20 = sub_15E0530(**(_QWORD **)(a1 + 40));
      v21 = *v19;
      v22 = v19[1];
      v36 = v20;
      v39 = v21;
      v40 = v22;
      if ( (_BYTE)v21 )
      {
        v17 = *(unsigned __int8 *)(v18 + (unsigned __int8)v21 + 1040);
      }
      else if ( sub_1F58D20((__int64)&v39) )
      {
        v44[0] = 0;
        v45 = 0;
        LOBYTE(v41[0]) = 0;
        v17 = sub_1F426C0(v18, v36, (unsigned int)v39, v40, (__int64)v44, (unsigned int *)&v42, v41);
      }
      else
      {
        v24 = sub_1F58D40((__int64)&v39);
        v41[0] = v21;
        v34 = v24;
        v41[1] = v22;
        if ( sub_1F58D20((__int64)v41) )
        {
          v44[0] = 0;
          v45 = 0;
          LOBYTE(v38) = 0;
          sub_1F426C0(v18, v36, LODWORD(v41[0]), v22, (__int64)v44, (unsigned int *)&v42, &v38);
          v26 = v38;
        }
        else
        {
          sub_1F40D10((__int64)v44, v18, v36, v21, v22);
          LOBYTE(v42) = v45;
          v43 = v46;
          if ( (_BYTE)v45 )
          {
            v26 = *(_BYTE *)(v18 + (unsigned __int8)v45 + 1155);
          }
          else
          {
            v31 = v46;
            if ( sub_1F58D20((__int64)&v42) )
            {
              v44[0] = 0;
              v45 = 0;
              v37 = 0;
              sub_1F426C0(v18, v36, (unsigned int)v42, v31, (__int64)v44, &v38, &v37);
              v26 = v37;
            }
            else
            {
              sub_1F40D10((__int64)v44, v18, v36, v42, v43);
              v25 = v30;
              LOBYTE(v25) = v45;
              v30 = v25;
              v26 = sub_1D5E9F0(v18, v36, (unsigned int)v25, v46);
            }
          }
        }
        v27 = sub_1FD3510(v26);
        v17 = (v27 + v34 - 1) / v27;
      }
      v35 += v17;
      v16 += 16;
    }
    while ( v33 != v16 );
  }
  sub_1FD5CC0(a1, a2, v35, 1);
  if ( (_BYTE *)v47[0] != v48 )
    _libc_free(v47[0]);
  return 1;
}
