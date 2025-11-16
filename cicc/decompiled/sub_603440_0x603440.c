// Function: sub_603440
// Address: 0x603440
//
unsigned int *__fastcall sub_603440(__int64 a1, __int64 *a2, __int64 a3)
{
  _BYTE *v3; // r13
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 *v8; // rcx
  int v9; // esi
  __int64 v10; // r9
  __int64 v11; // rbx
  __int64 *v12; // r13
  _BOOL4 v13; // r12d
  char *v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r14
  int v18; // eax
  __int64 v19; // rax
  char i; // dl
  __int64 v21; // rax
  __int64 v23; // r12
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned int v28; // [rsp+14h] [rbp-CCh]
  int v30; // [rsp+2Ch] [rbp-B4h] BYREF
  __int64 v31; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-A8h]
  char v33; // [rsp+3Ch] [rbp-A4h]
  __int64 v34; // [rsp+40h] [rbp-A0h]
  __int64 v35; // [rsp+48h] [rbp-98h]
  __int64 v36; // [rsp+50h] [rbp-90h]
  __int64 v37; // [rsp+58h] [rbp-88h]
  __int64 v38; // [rsp+60h] [rbp-80h]
  __int64 v39[14]; // [rsp+70h] [rbp-70h] BYREF

  v3 = (_BYTE *)a1;
  v5 = *(_QWORD *)(a1 + 168);
  v31 = a1;
  v33 = 0;
  v34 = 0;
  v28 = dword_4F04C3C;
  v35 = 0;
  dword_4F04C3C = 1;
  v36 = 0;
  v38 = 0;
  v37 = 0;
  v32 = v32 & 0xF8000000 | 1;
  v6 = sub_8600D0(6, 0xFFFFFFFFLL, a1, 0);
  v8 = &v31;
  *(_QWORD *)(v5 + 152) = v6;
  v9 = dword_4F04C64;
  *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 600) = &v31;
  v10 = *a2;
  v11 = *a2 + 32 * a2[2];
  if ( v10 == v11 )
    goto LABEL_19;
  v12 = (__int64 *)v10;
  v13 = 1;
  do
  {
    while ( 1 )
    {
      v14 = (char *)v12[1];
      v15 = *v12;
      v17 = sub_5F7E50(v14, *v12);
      if ( v17 )
      {
        LOBYTE(v16) = *(_BYTE *)v12[1] == 0;
        v30 = (unsigned __int8)v16;
        v18 = *((_DWORD *)v12 + 4);
        if ( v18 )
          *(_DWORD *)(v17 + 140) = v18;
        if ( v12[3] || (_BYTE)v16 )
        {
          v39[0] = sub_724DC0(v14, v15, v16, v8, v7, v10);
          sub_72BAF0(v39[0], v12[3], 5);
          *(_BYTE *)(v17 + 144) |= 4u;
          sub_5E53A0(v17, v39[0], &v30, (__int64 *)(v17 + 120), a3);
          sub_724E30(v39);
        }
      }
      if ( v13 )
      {
        v19 = sub_8D4130(*v12);
        for ( i = *(_BYTE *)(v19 + 140); i == 12; i = *(_BYTE *)(v19 + 140) )
          v19 = *(_QWORD *)(v19 + 160);
        if ( (unsigned __int8)(i - 9) <= 2u )
        {
          v21 = *(_QWORD *)(*(_QWORD *)v19 + 96LL);
          if ( *(_QWORD *)(v21 + 24) )
            break;
        }
      }
      v12 += 4;
      if ( (__int64 *)v11 == v12 )
        goto LABEL_16;
    }
    v12 += 4;
    v13 = (*(_BYTE *)(v21 + 177) & 2) != 0;
  }
  while ( (__int64 *)v11 != v12 );
LABEL_16:
  v3 = (_BYTE *)a1;
  if ( *(_BYTE *)(a1 + 140) != 11 || v13 )
  {
    v9 = dword_4F04C64;
LABEL_19:
    sub_601910(v3, (unsigned int)(v9 - 1), (__int64)&v31, (__int64)v8, v7, v10);
    *(_BYTE *)(*(_QWORD *)v3 + 81LL) |= 2u;
    if ( (v3[177] & 0x10) != 0 )
      v3[178] |= 1u;
    sub_863FC0();
    goto LABEL_22;
  }
  memset(v39, 0, 64);
  v23 = sub_5FE7E0(&v31, (__int64)v39);
  sub_601910((_BYTE *)a1, (unsigned int)(dword_4F04C64 - 1), (__int64)&v31, v24, v25, v26);
  *(_BYTE *)(*(_QWORD *)a1 + 81LL) |= 2u;
  if ( (*(_BYTE *)(a1 + 177) & 0x10) != 0 )
    *(_BYTE *)(a1 + 178) |= 1u;
  sub_863FC0();
  if ( v23 )
    sub_71D150(v23);
LABEL_22:
  dword_4F04C3C = v28;
  return &dword_4F04C3C;
}
