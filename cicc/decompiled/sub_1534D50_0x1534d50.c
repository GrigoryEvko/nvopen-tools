// Function: sub_1534D50
// Address: 0x1534d50
//
void __fastcall sub_1534D50(_DWORD *a1, __int64 a2, _QWORD *a3)
{
  unsigned __int64 **v3; // rax
  __int64 v5; // rdx
  unsigned __int64 *v6; // r14
  __int64 v7; // rax
  unsigned __int64 *i; // rcx
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 *v11; // rsi
  unsigned __int64 *v12; // r8
  unsigned __int64 *v13; // r14
  __int64 j; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  unsigned __int64 *v17; // rsi
  unsigned __int64 *v18; // r14
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  unsigned __int64 *v22; // r14
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rcx
  unsigned __int64 *v26; // r15
  __int64 v27; // rax
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  _BOOL4 v30; // r12d
  _QWORD *v31; // [rsp+8h] [rbp-258h]
  unsigned __int64 *v33; // [rsp+18h] [rbp-248h]
  unsigned __int64 *v34; // [rsp+18h] [rbp-248h]
  unsigned __int64 *v35; // [rsp+18h] [rbp-248h]
  unsigned __int64 *v36; // [rsp+18h] [rbp-248h]
  unsigned __int64 *v37; // [rsp+18h] [rbp-248h]
  unsigned __int64 *v38; // [rsp+20h] [rbp-240h] BYREF
  __int64 v39; // [rsp+28h] [rbp-238h]
  _BYTE v40[560]; // [rsp+30h] [rbp-230h] BYREF

  v3 = *(unsigned __int64 ***)(a2 + 96);
  if ( !v3 )
    goto LABEL_49;
  v5 = v3[1] - *v3;
  if ( !v5 )
  {
LABEL_3:
    v38 = (unsigned __int64 *)v40;
    v39 = 0x4000000000LL;
LABEL_4:
    v6 = v3[3];
    v33 = v3[4];
    if ( v6 == v33 )
      goto LABEL_13;
    v7 = 0;
    for ( i = (unsigned __int64 *)v40; ; i = v38 )
    {
      i[v7] = *v6;
      v9 = (unsigned int)(v39 + 1);
      LODWORD(v39) = v9;
      if ( HIDWORD(v39) <= (unsigned int)v9 )
      {
        sub_16CD150(&v38, v40, 0, 8);
        v9 = (unsigned int)v39;
      }
      v10 = v6[1];
      v11 = v6;
      v6 += 2;
      v38[v9] = v10;
      LODWORD(v39) = v39 + 1;
      sub_A19EB0(a3, v11);
      if ( v6 == v33 )
        break;
      v7 = (unsigned int)v39;
      if ( (unsigned int)v39 >= HIDWORD(v39) )
      {
        sub_16CD150(&v38, v40, 0, 8);
        v7 = (unsigned int)v39;
      }
    }
    sub_152F3D0(a1, 0xCu, (__int64)&v38, 0);
    v3 = *(unsigned __int64 ***)(a2 + 96);
    if ( v3 )
    {
LABEL_13:
      v12 = v3[6];
      v34 = v3[7];
      if ( v12 == v34 )
        goto LABEL_22;
      LODWORD(v39) = 0;
      v13 = v12;
      for ( j = 0; ; j = (unsigned int)v39 )
      {
        if ( HIDWORD(v39) <= (unsigned int)j )
        {
          sub_16CD150(&v38, v40, 0, 8);
          j = (unsigned int)v39;
        }
        v38[j] = *v13;
        v15 = (unsigned int)(v39 + 1);
        LODWORD(v39) = v15;
        if ( HIDWORD(v39) <= (unsigned int)v15 )
        {
          sub_16CD150(&v38, v40, 0, 8);
          v15 = (unsigned int)v39;
        }
        v16 = v13[1];
        v17 = v13;
        v13 += 2;
        v38[v15] = v16;
        LODWORD(v39) = v39 + 1;
        sub_A19EB0(a3, v17);
        if ( v34 == v13 )
          break;
      }
      sub_152F3D0(a1, 0xDu, (__int64)&v38, 0);
      v3 = *(unsigned __int64 ***)(a2 + 96);
      if ( v3 )
      {
LABEL_22:
        v35 = v3[10];
        if ( v3[9] == v35 )
          goto LABEL_30;
        v18 = v3[9];
        do
        {
          v19 = 0;
          LODWORD(v39) = 0;
          if ( !HIDWORD(v39) )
          {
            sub_16CD150(&v38, v40, 0, 8);
            v19 = (unsigned int)v39;
          }
          v38[v19] = *v18;
          LODWORD(v39) = v39 + 1;
          sub_A19EB0(a3, v18);
          v20 = (unsigned int)v39;
          if ( (unsigned int)v39 >= HIDWORD(v39) )
          {
            sub_16CD150(&v38, v40, 0, 8);
            v20 = (unsigned int)v39;
          }
          v21 = v18[1];
          v18 += 5;
          v38[v20] = v21;
          LODWORD(v39) = v39 + 1;
          sub_1523C30((__int64)&v38, (char *)&v38[(unsigned int)v39], (char *)*(v18 - 3), (char *)*(v18 - 2));
          sub_152F3D0(a1, 0xEu, (__int64)&v38, 0);
        }
        while ( v35 != v18 );
        v3 = *(unsigned __int64 ***)(a2 + 96);
        if ( v3 )
        {
LABEL_30:
          v36 = v3[13];
          if ( v3[12] != v36 )
          {
            v22 = v3[12];
            do
            {
              v23 = 0;
              LODWORD(v39) = 0;
              if ( !HIDWORD(v39) )
              {
                sub_16CD150(&v38, v40, 0, 8);
                v23 = (unsigned int)v39;
              }
              v38[v23] = *v22;
              LODWORD(v39) = v39 + 1;
              sub_A19EB0(a3, v22);
              v24 = (unsigned int)v39;
              if ( (unsigned int)v39 >= HIDWORD(v39) )
              {
                sub_16CD150(&v38, v40, 0, 8);
                v24 = (unsigned int)v39;
              }
              v25 = v22[1];
              v22 += 5;
              v38[v24] = v25;
              LODWORD(v39) = v39 + 1;
              sub_1523C30((__int64)&v38, (char *)&v38[(unsigned int)v39], (char *)*(v22 - 3), (char *)*(v22 - 2));
              sub_152F3D0(a1, 0xFu, (__int64)&v38, 0);
            }
            while ( v36 != v22 );
          }
        }
      }
    }
    goto LABEL_37;
  }
  v38 = *v3;
  v39 = v5;
  sub_152A900(a1, 0xBu, (__int64 *)&v38, 0);
  v3 = *(unsigned __int64 ***)(a2 + 96);
  if ( !v3 )
  {
LABEL_49:
    v38 = (unsigned __int64 *)v40;
    goto LABEL_37;
  }
  v26 = *v3;
  v37 = v3[1];
  if ( *v3 == v37 )
    goto LABEL_3;
  do
  {
    v28 = sub_1534CB0((__int64)a3, v26);
    if ( v29 )
    {
      v30 = 1;
      if ( !v28 && v29 != a3 + 1 )
        v30 = *v26 < v29[4];
      v31 = v29;
      v27 = sub_22077B0(40);
      *(_QWORD *)(v27 + 32) = *v26;
      sub_220F040(v30, v27, v31, a3 + 1);
      ++a3[5];
    }
    ++v26;
  }
  while ( v37 != v26 );
  v38 = (unsigned __int64 *)v40;
  v3 = *(unsigned __int64 ***)(a2 + 96);
  v39 = 0x4000000000LL;
  if ( v3 )
    goto LABEL_4;
LABEL_37:
  if ( v38 != (unsigned __int64 *)v40 )
    _libc_free((unsigned __int64)v38);
}
