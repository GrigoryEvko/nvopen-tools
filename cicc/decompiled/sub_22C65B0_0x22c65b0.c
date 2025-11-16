// Function: sub_22C65B0
// Address: 0x22c65b0
//
__int64 __fastcall sub_22C65B0(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned int v5; // eax
  unsigned int v6; // r12d
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 *v10; // rax
  __int64 v11; // r14
  unsigned __int8 *v12; // rdi
  char v13; // dl
  __int64 v14; // rcx
  __int64 *v15; // rax
  __int64 *v16; // r8
  __int64 v17; // rdx
  char v18; // dl
  __int64 v19; // r14
  __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 j; // rcx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rdx
  bool v28; // dl
  __int64 v29; // rsi
  int v30; // edi
  unsigned int v31; // eax
  unsigned __int8 *v32; // rcx
  int v33; // eax
  __int64 v34; // rcx
  __int64 v35; // rax
  int v36; // r8d
  unsigned __int8 *v37; // [rsp+8h] [rbp-C8h]
  __int64 *v38; // [rsp+10h] [rbp-C0h]
  __int64 v39; // [rsp+10h] [rbp-C0h]
  __int64 v41; // [rsp+18h] [rbp-B8h]
  __int64 *v42; // [rsp+18h] [rbp-B8h]
  __int64 v43; // [rsp+18h] [rbp-B8h]
  __int64 v44; // [rsp+18h] [rbp-B8h]
  __int64 v45; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+28h] [rbp-A8h]
  __int64 v47; // [rsp+30h] [rbp-A0h]
  __int64 v48; // [rsp+40h] [rbp-90h] BYREF
  __int64 v49; // [rsp+48h] [rbp-88h]
  __int64 v50; // [rsp+50h] [rbp-80h]
  __int64 v51; // [rsp+60h] [rbp-70h] BYREF
  __int64 v52; // [rsp+68h] [rbp-68h]
  __int64 i; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v54; // [rsp+78h] [rbp-58h]
  _BYTE v55[48]; // [rsp+A0h] [rbp-30h] BYREF

  v4 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  LOBYTE(v5) = sub_B2F070(*(_QWORD *)(a3 + 72), *(_DWORD *)(v4 + 8) >> 8);
  v6 = v5;
  if ( (_BYTE)v5 )
    return 0;
  v37 = sub_BD4CB0((unsigned __int8 *)a2, (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_96, (__int64)&v51);
  v8 = sub_22C23A0(a1, a3);
  if ( !*(_BYTE *)(v8 + 448) )
  {
    v51 = 0;
    v9 = a3;
    v10 = &i;
    v52 = 1;
    do
    {
      *v10 = 0;
      v10 += 3;
      *(v10 - 2) = 0;
      *(v10 - 1) = -4096;
    }
    while ( v10 != (__int64 *)v55 );
    v11 = *(_QWORD *)(a3 + 56);
    v41 = a3 + 48;
    if ( v11 != v9 + 48 )
    {
      do
      {
        v12 = (unsigned __int8 *)(v11 - 24);
        if ( !v11 )
          v12 = 0;
        sub_22C6290(v12, (__int64)&v51);
        v11 = *(_QWORD *)(v11 + 8);
      }
      while ( v41 != v11 );
    }
    if ( *(_BYTE *)(v8 + 448) )
    {
      sub_22C1AA0(v8 + 384);
      if ( (*(_BYTE *)(v8 + 392) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v8 + 400), 24LL * *(unsigned int *)(v8 + 408), 8);
      *(_BYTE *)(v8 + 392) |= 1u;
      sub_22C3BD0(v8 + 384);
      sub_22C3280(v8 + 384, (__int64)&v51);
    }
    else
    {
      *(_BYTE *)(v8 + 392) |= 1u;
      *(_QWORD *)(v8 + 384) = 0;
      sub_22C3BD0(v8 + 384);
      sub_22C3280(v8 + 384, (__int64)&v51);
      *(_BYTE *)(v8 + 448) = 1;
    }
    v13 = v52;
    if ( (v52 & 1) != 0 )
    {
      v45 = 0;
      v16 = (__int64 *)v55;
      v15 = &i;
      v46 = 0;
      v47 = -4096;
      v48 = 0;
      v49 = 0;
      v50 = -8192;
    }
    else
    {
      v14 = v54;
      if ( !v54 )
        goto LABEL_56;
      v15 = (__int64 *)i;
      v45 = 0;
      v46 = 0;
      v16 = (__int64 *)(i + 24LL * v54);
      v47 = -4096;
      v48 = 0;
      v49 = 0;
      v50 = -8192;
      if ( v16 == (__int64 *)i )
      {
LABEL_24:
        if ( (v13 & 1) != 0 )
        {
LABEL_25:
          v18 = *(_BYTE *)(v8 + 392) & 1;
          if ( *(_DWORD *)(v8 + 392) >> 1 )
          {
            if ( v18 )
            {
              v19 = v8 + 400;
              v20 = 48;
            }
            else
            {
              v19 = *(_QWORD *)(v8 + 400);
              v20 = 24LL * *(unsigned int *)(v8 + 408);
            }
            v21 = v19 + v20;
            v50 = -4096;
            v48 = 0;
            v49 = 0;
            v51 = 0;
            v52 = 0;
            for ( i = -8192; v21 != v19; v19 += 24 )
            {
              v22 = *(_QWORD *)(v19 + 16);
              if ( v22 != -4096 && v22 != -8192 )
                break;
            }
            v43 = v21;
            sub_D68D70(&v51);
            sub_D68D70(&v48);
            j = v43;
            v18 = *(_BYTE *)(v8 + 392) & 1;
          }
          else
          {
            if ( v18 )
            {
              v34 = v8 + 400;
              v35 = 48;
            }
            else
            {
              v34 = *(_QWORD *)(v8 + 400);
              v35 = 24LL * *(unsigned int *)(v8 + 408);
            }
            v19 = v34 + v35;
            j = v34 + v35;
          }
          if ( v18 )
          {
            v24 = v8 + 400;
            v25 = 48;
          }
          else
          {
            v24 = *(_QWORD *)(v8 + 400);
            v25 = 24LL * *(unsigned int *)(v8 + 408);
          }
          v39 = v24 + v25;
          while ( v39 != v19 )
          {
            v26 = *(_QWORD *)(v19 + 16);
            v44 = j;
            v19 += 24;
            sub_22BECA0(a1, v26);
            for ( j = v44; v19 != v44; v19 += 24 )
            {
              v27 = *(_QWORD *)(v19 + 16);
              if ( v27 != -4096 && v27 != -8192 )
                break;
            }
          }
          goto LABEL_41;
        }
        v14 = v54;
LABEL_56:
        sub_C7D6A0(i, 24 * v14, 8);
        goto LABEL_25;
      }
    }
    do
    {
      v17 = v15[2];
      if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
      {
        v38 = v16;
        v42 = v15;
        sub_BD60C0(v15);
        v16 = v38;
        v15 = v42;
      }
      v15 += 3;
    }
    while ( v15 != v16 );
    v13 = v52;
    if ( v47 != -8192 && v47 != 0 && v47 != -4096 )
    {
      sub_BD60C0(&v45);
      v13 = v52;
    }
    goto LABEL_24;
  }
LABEL_41:
  v51 = 0;
  v52 = 0;
  i = (__int64)v37;
  v28 = v37 + 4096 != 0 && v37 != 0 && v37 + 0x2000 != 0;
  if ( v28 )
  {
    sub_BD73F0((__int64)&v51);
    v37 = (unsigned __int8 *)i;
    v28 = i != -4096 && i != -8192 && i != 0;
  }
  if ( (*(_BYTE *)(v8 + 392) & 1) != 0 )
  {
    v29 = v8 + 400;
    v30 = 1;
    goto LABEL_45;
  }
  v33 = *(_DWORD *)(v8 + 408);
  v29 = *(_QWORD *)(v8 + 400);
  v30 = v33 - 1;
  if ( v33 )
  {
LABEL_45:
    v31 = v30 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
    v32 = *(unsigned __int8 **)(v29 + 24LL * v31 + 16);
    if ( v32 == v37 )
    {
LABEL_46:
      v6 = 1;
    }
    else
    {
      v36 = 1;
      while ( v32 != (unsigned __int8 *)-4096LL )
      {
        v31 = v30 & (v36 + v31);
        v32 = *(unsigned __int8 **)(v29 + 24LL * v31 + 16);
        if ( v32 == v37 )
          goto LABEL_46;
        ++v36;
      }
    }
  }
  if ( v28 )
    sub_BD60C0(&v51);
  return v6;
}
