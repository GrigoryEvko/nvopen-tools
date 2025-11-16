// Function: sub_27CE9A0
// Address: 0x27ce9a0
//
__int64 __fastcall sub_27CE9A0(unsigned __int64 *a1, int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // r12
  int v11; // ecx
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // ecx
  _QWORD *v16; // rdx
  _QWORD *v17; // r13
  unsigned __int64 v18; // r14
  __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned int i; // eax
  __int64 v22; // rsi
  int v23; // eax
  __int64 v24; // rbx
  __int64 *v25; // r12
  int v26; // edx
  _QWORD *v27; // rax
  unsigned __int64 *v28; // r12
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 v35; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  unsigned __int64 v37[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v38; // [rsp+30h] [rbp-50h]
  __int16 v39; // [rsp+40h] [rbp-40h]

  v6 = *a1;
  v7 = *(_QWORD *)(*a1 + 8);
  v10 = (__int64 *)sub_BCE3C0(*(__int64 **)v7, a2);
  v11 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
  {
    BYTE4(v35) = (_BYTE)v11 == 18;
    LODWORD(v35) = *(_DWORD *)(v7 + 32);
    v10 = (__int64 *)sub_BCE1B0(v10, v35);
  }
  if ( *(_BYTE *)v6 <= 0x15u )
    return sub_ADA8A0(v6, (__int64)v10, 0);
  v13 = *(unsigned int *)(a3 + 24);
  if ( (_DWORD)v13 )
  {
    v14 = *(_QWORD *)(a3 + 8);
    v15 = (v13 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v16 = (_QWORD *)(v14 + ((unsigned __int64)v15 << 6));
    v8 = v16[3];
    if ( v6 == v8 )
    {
LABEL_7:
      if ( v16 != (_QWORD *)(v14 + (v13 << 6)) )
      {
        v37[0] = 6;
        v37[1] = 0;
        v38 = (_QWORD *)v16[7];
        v17 = v38;
        if ( v38 + 512 != 0 && v38 != 0 && v38 != (_QWORD *)-8192LL )
        {
          sub_BD6050(v37, v16[5] & 0xFFFFFFFFFFFFFFF8LL);
          v17 = v38;
        }
        if ( v17 )
        {
          if ( v17 != (_QWORD *)-4096LL && v17 != (_QWORD *)-8192LL )
            sub_BD60C0(v37);
          return (__int64)v17;
        }
      }
    }
    else
    {
      v29 = 1;
      while ( v8 != -4096 )
      {
        v9 = (unsigned int)(v29 + 1);
        v15 = (v13 - 1) & (v29 + v15);
        v16 = (_QWORD *)(v14 + ((unsigned __int64)v15 << 6));
        v8 = v16[3];
        if ( v6 == v8 )
          goto LABEL_7;
        v29 = v9;
      }
    }
  }
  v18 = a1[3];
  v19 = *(unsigned int *)(a4 + 24);
  v20 = *(_QWORD *)(a4 + 8);
  if ( (_DWORD)v19 )
  {
    v9 = 1;
    for ( i = (v19 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)
                | ((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)))); ; i = (v19 - 1) & v23 )
    {
      v22 = v20 + 24LL * i;
      v8 = *(_QWORD *)v22;
      if ( v18 == *(_QWORD *)v22 && v6 == *(_QWORD *)(v22 + 8) )
        break;
      if ( v8 == -4096 && *(_QWORD *)(v22 + 8) == -4096 )
        goto LABEL_36;
      v23 = v9 + i;
      v9 = (unsigned int)(v9 + 1);
    }
    if ( v22 != v20 + 24 * v19 )
    {
      v24 = *(_QWORD *)(v6 + 8);
      v25 = (__int64 *)sub_BCE3C0(*(__int64 **)v24, *(_DWORD *)(v22 + 16));
      v26 = *(unsigned __int8 *)(v24 + 8);
      if ( (unsigned int)(v26 - 17) <= 1 )
      {
        BYTE4(v36) = (_BYTE)v26 == 18;
        LODWORD(v36) = *(_DWORD *)(v24 + 32);
        v25 = (__int64 *)sub_BCE1B0(v25, v36);
      }
      v39 = 257;
      v27 = sub_BD2C40(72, unk_3F10A14);
      v17 = v27;
      if ( v27 )
        sub_B51C90((__int64)v27, v6, (__int64)v25, (__int64)v37, 0, 0);
      sub_B44220(v17, v18 + 24, 0);
      v28 = v17 + 6;
      v37[0] = *(_QWORD *)(v18 + 48);
      if ( v37[0] )
      {
        sub_B96E90((__int64)v37, v37[0], 1);
        if ( v28 == v37 )
        {
          if ( v37[0] )
            sub_B91220((__int64)v37, v37[0]);
          return (__int64)v17;
        }
        v31 = v17[6];
        if ( !v31 )
        {
LABEL_42:
          v32 = (unsigned __int8 *)v37[0];
          v17[6] = v37[0];
          if ( v32 )
            sub_B976B0((__int64)v37, v32, (__int64)(v17 + 6));
          return (__int64)v17;
        }
      }
      else
      {
        if ( v28 == v37 )
          return (__int64)v17;
        v31 = v17[6];
        if ( !v31 )
          return (__int64)v17;
      }
      sub_B91220((__int64)(v17 + 6), v31);
      goto LABEL_42;
    }
  }
LABEL_36:
  v30 = *(unsigned int *)(a5 + 8);
  if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
  {
    sub_C8D5F0(a5, (const void *)(a5 + 16), v30 + 1, 8u, v8, v9);
    v30 = *(unsigned int *)(a5 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a5 + 8 * v30) = a1;
  ++*(_DWORD *)(a5 + 8);
  return sub_ACADE0((__int64 **)v10);
}
