// Function: sub_1D69330
// Address: 0x1d69330
//
__int64 __fastcall sub_1D69330(__int64 a1)
{
  unsigned int v1; // r15d
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rdi
  _QWORD *v5; // rcx
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  _QWORD *v14; // r14
  _QWORD *v15; // rax
  int v16; // edx
  unsigned __int64 v17; // rdx
  __int64 v19; // rax
  int v20; // r10d
  __int64 *v21; // r15
  int v22; // eax
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rsi
  int v31; // r10d
  __int64 *v32; // r8
  __int64 v33; // [rsp+10h] [rbp-A0h]
  __int64 v35; // [rsp+38h] [rbp-78h] BYREF
  __int64 v36[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v37; // [rsp+50h] [rbp-60h]
  __int64 v38; // [rsp+60h] [rbp-50h] BYREF
  __int64 v39; // [rsp+68h] [rbp-48h]
  __int64 v40; // [rsp+70h] [rbp-40h]
  unsigned int v41; // [rsp+78h] [rbp-38h]

  v1 = 0;
  v2 = 0x40018000000001LL;
  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_QWORD *)(a1 + 40);
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  if ( !v3 )
    goto LABEL_29;
  do
  {
    while ( 1 )
    {
      v14 = (_QWORD *)v3;
      v15 = sub_1648700(v3);
      v3 = *(_QWORD *)(v3 + 8);
      v6 = v15[5];
      v16 = *((unsigned __int8 *)v15 + 16);
      v35 = v6;
      if ( (_BYTE)v16 != 77 )
        break;
      if ( (*((_BYTE *)v15 + 23) & 0x40) != 0 )
        v5 = (_QWORD *)*(v15 - 1);
      else
        v5 = &v15[-3 * (*((_DWORD *)v15 + 5) & 0xFFFFFFF)];
      v6 = v5[3 * *((unsigned int *)v15 + 14) + 1 + -1431655765 * (unsigned int)(v14 - v5)];
      v35 = v6;
LABEL_6:
      v7 = (unsigned int)*(unsigned __int8 *)(sub_157EBA0(v6) + 16) - 34;
      if ( (unsigned int)v7 <= 0x36 && _bittest64(&v2, v7) || v4 == v6 )
        goto LABEL_17;
      if ( !v41 )
      {
        ++v38;
        goto LABEL_54;
      }
      v8 = (v41 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v9 = (__int64 *)(v39 + 16LL * v8);
      v10 = *v9;
      if ( *v9 != v6 )
      {
        v20 = 1;
        v21 = 0;
        while ( v10 != -8 )
        {
          if ( v10 == -16 && !v21 )
            v21 = v9;
          v8 = (v41 - 1) & (v20 + v8);
          v9 = (__int64 *)(v39 + 16LL * v8);
          v10 = *v9;
          if ( *v9 == v6 )
            goto LABEL_11;
          ++v20;
        }
        if ( !v21 )
          v21 = v9;
        ++v38;
        v22 = v40 + 1;
        if ( 4 * ((int)v40 + 1) >= 3 * v41 )
        {
LABEL_54:
          sub_1D69170((__int64)&v38, 2 * v41);
          if ( !v41 )
          {
            LODWORD(v40) = v40 + 1;
            BUG();
          }
          v6 = v35;
          LODWORD(v29) = (v41 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
          v21 = (__int64 *)(v39 + 16LL * (unsigned int)v29);
          v30 = *v21;
          if ( *v21 == v35 )
          {
LABEL_56:
            v22 = v40 + 1;
          }
          else
          {
            v31 = 1;
            v32 = 0;
            while ( v30 != -8 )
            {
              if ( v30 == -16 && !v32 )
                v32 = v21;
              v29 = (v41 - 1) & ((_DWORD)v29 + v31);
              v21 = (__int64 *)(v39 + 16 * v29);
              v30 = *v21;
              if ( v35 == *v21 )
                goto LABEL_56;
              ++v31;
            }
            v22 = v40 + 1;
            if ( v32 )
              v21 = v32;
          }
        }
        else if ( v41 - HIDWORD(v40) - v22 <= v41 >> 3 )
        {
          sub_1D69170((__int64)&v38, v41);
          sub_1D67D20((__int64)&v38, &v35, v36);
          v21 = (__int64 *)v36[0];
          v6 = v35;
          v22 = v40 + 1;
        }
        LODWORD(v40) = v22;
        if ( *v21 != -8 )
          --HIDWORD(v40);
        *v21 = v6;
        v21[1] = 0;
        goto LABEL_40;
      }
LABEL_11:
      v11 = v9[1];
      if ( v11 )
      {
        v12 = v3;
        if ( !*v14 )
        {
          *v14 = v11;
LABEL_26:
          v19 = *(_QWORD *)(v11 + 8);
          v14[1] = v19;
          if ( v19 )
            *(_QWORD *)(v19 + 16) = (unsigned __int64)(v14 + 1) | *(_QWORD *)(v19 + 16) & 3LL;
          v14[2] = (v11 + 8) | v14[2] & 3LL;
          *(_QWORD *)(v11 + 8) = v14;
          goto LABEL_16;
        }
        goto LABEL_13;
      }
      v21 = v9;
LABEL_40:
      v23 = sub_157EE30(v35);
      if ( v23 )
        v23 -= 24;
      v37 = 257;
      v24 = sub_15FDBD0(
              (unsigned int)*(unsigned __int8 *)(a1 + 16) - 24,
              *(_QWORD *)(a1 - 24),
              *(_QWORD *)a1,
              (__int64)v36,
              v23);
      v21[1] = v24;
      v25 = v24;
      v36[0] = *(_QWORD *)(a1 + 48);
      if ( v36[0] )
      {
        sub_1623A60((__int64)v36, v36[0], 2);
        v26 = v25 + 48;
        if ( (__int64 *)(v25 + 48) != v36 )
        {
          v27 = *(_QWORD *)(v25 + 48);
          if ( v27 )
          {
LABEL_50:
            v33 = v26;
            sub_161E7C0(v26, v27);
            v26 = v33;
          }
          v28 = (unsigned __int8 *)v36[0];
          *(_QWORD *)(v25 + 48) = v36[0];
          if ( v28 )
            sub_1623210((__int64)v36, v28, v26);
          goto LABEL_46;
        }
        if ( v36[0] )
          sub_161E7C0((__int64)v36, v36[0]);
      }
      else
      {
        v26 = v24 + 48;
        if ( (__int64 *)(v24 + 48) != v36 )
        {
          v27 = *(_QWORD *)(v24 + 48);
          if ( v27 )
            goto LABEL_50;
        }
      }
LABEL_46:
      v11 = v21[1];
      if ( !*v14 )
        goto LABEL_15;
      v12 = v14[1];
LABEL_13:
      v13 = v14[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v13 = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
LABEL_15:
      *v14 = v11;
      if ( v11 )
        goto LABEL_26;
LABEL_16:
      v1 = 1;
LABEL_17:
      if ( !v3 )
        goto LABEL_22;
    }
    v17 = (unsigned int)(v16 - 34);
    if ( (unsigned int)v17 > 0x36 || !_bittest64(&v2, v17) )
      goto LABEL_6;
  }
  while ( v3 );
LABEL_22:
  if ( !*(_QWORD *)(a1 + 8) )
  {
LABEL_29:
    v1 = 1;
    sub_1AEAA40(a1);
    sub_15F20C0((_QWORD *)a1);
  }
  j___libc_free_0(v39);
  return v1;
}
