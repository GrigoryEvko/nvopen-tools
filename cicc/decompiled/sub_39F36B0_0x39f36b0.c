// Function: sub_39F36B0
// Address: 0x39f36b0
//
void __fastcall sub_39F36B0(__int64 *a1)
{
  _QWORD *v2; // rdi
  __int64 *v3; // r15
  __int64 *v4; // rbx
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rcx
  unsigned int v11; // esi
  unsigned int v12; // edi
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // r8
  __int64 v15; // r11
  __int64 v16; // rbx
  unsigned int v17; // r8d
  unsigned __int64 v18; // r10
  unsigned int v19; // r9d
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 i; // rdi
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r14
  __int64 v26; // rdx
  int v27; // edx
  int v28; // r12d
  unsigned int v29; // edx
  int v30; // edi
  unsigned __int64 v31; // r14
  unsigned __int64 *v32; // r11
  int v33; // r8d
  unsigned int v34; // edx
  unsigned __int64 *v35; // rsi
  unsigned __int64 v36; // r9
  int v37; // r8d
  unsigned __int64 v38; // [rsp+10h] [rbp-60h]
  int v39; // [rsp+10h] [rbp-60h]
  unsigned __int64 v40; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v42; // [rsp+28h] [rbp-48h]
  __int64 v43; // [rsp+30h] [rbp-40h]
  unsigned int v44; // [rsp+38h] [rbp-38h]

  sub_38D4AB0((__int64)a1, *(_QWORD *)(a1[33] + 8));
  v2 = (_QWORD *)a1[33];
  v41 = 0;
  v42 = 0;
  v3 = (__int64 *)v2[8];
  v4 = (__int64 *)v2[7];
  v43 = 0;
  v44 = 0;
  while ( v3 != v4 )
  {
    while ( 1 )
    {
      v5 = *v4;
      if ( !sub_390B160((__int64)v2, *v4) )
        goto LABEL_3;
      v6 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v6 )
      {
        if ( (*(_BYTE *)(v5 + 9) & 0xC) != 8 )
          goto LABEL_3;
        *(_BYTE *)(v5 + 8) |= 4u;
        v7 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v5 + 24));
        v8 = v7 | *(_QWORD *)v5 & 7LL;
        *(_QWORD *)v5 = v8;
        if ( !v7 )
          goto LABEL_3;
        v9 = v8 & 0xFFFFFFFFFFFFFFF8LL;
        v6 = v9;
        if ( !v9 )
          break;
      }
LABEL_29:
      if ( off_4CF6DB8 == (_UNKNOWN *)v6 || (*(_BYTE *)(v5 + 9) & 0xC) == 8 )
        goto LABEL_3;
      v11 = v44;
      v9 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = v9;
      if ( !v44 )
      {
LABEL_32:
        ++v41;
        goto LABEL_33;
      }
LABEL_12:
      v12 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v13 = (unsigned __int64 *)(v42 + 16LL * v12);
      v14 = *v13;
      if ( v10 != *v13 )
      {
        v39 = 1;
        v32 = 0;
        while ( v14 != -8 )
        {
          if ( v14 == -16 && !v32 )
            v32 = v13;
          v12 = (v11 - 1) & (v39 + v12);
          v13 = (unsigned __int64 *)(v42 + 16LL * v12);
          v14 = *v13;
          if ( v10 == *v13 )
            goto LABEL_13;
          ++v39;
        }
        if ( v32 )
          v13 = v32;
        ++v41;
        v30 = v43 + 1;
        if ( 4 * ((int)v43 + 1) >= 3 * v11 )
        {
LABEL_33:
          v38 = v10;
          sub_39F34F0((__int64)&v41, 2 * v11);
          if ( !v44 )
            goto LABEL_67;
          v10 = v38;
          v29 = (v44 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v30 = v43 + 1;
          v13 = (unsigned __int64 *)(v42 + 16LL * v29);
          v31 = *v13;
          if ( v38 != *v13 )
          {
            v37 = 1;
            v35 = 0;
            while ( v31 != -8 )
            {
              if ( v31 == -16 && !v35 )
                v35 = v13;
              v29 = (v44 - 1) & (v37 + v29);
              v13 = (unsigned __int64 *)(v42 + 16LL * v29);
              v31 = *v13;
              if ( v38 == *v13 )
                goto LABEL_35;
              ++v37;
            }
LABEL_47:
            if ( v35 )
              v13 = v35;
          }
        }
        else if ( v11 - HIDWORD(v43) - v30 <= v11 >> 3 )
        {
          v40 = v10;
          sub_39F34F0((__int64)&v41, v11);
          if ( !v44 )
          {
LABEL_67:
            LODWORD(v43) = v43 + 1;
            BUG();
          }
          v33 = 1;
          v10 = v40;
          v34 = (v44 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v30 = v43 + 1;
          v35 = 0;
          v13 = (unsigned __int64 *)(v42 + 16LL * v34);
          v36 = *v13;
          if ( v40 != *v13 )
          {
            while ( v36 != -8 )
            {
              if ( v36 == -16 && !v35 )
                v35 = v13;
              v34 = (v44 - 1) & (v33 + v34);
              v13 = (unsigned __int64 *)(v42 + 16LL * v34);
              v36 = *v13;
              if ( v40 == *v13 )
                goto LABEL_35;
              ++v33;
            }
            goto LABEL_47;
          }
        }
LABEL_35:
        LODWORD(v43) = v30;
        if ( *v13 != -8 )
          --HIDWORD(v43);
        *v13 = v10;
        v13[1] = 0;
      }
LABEL_13:
      ++v4;
      v13[1] = v5;
      v2 = (_QWORD *)a1[33];
      if ( v3 == v4 )
        goto LABEL_14;
    }
    if ( (*(_BYTE *)(v5 + 9) & 0xC) == 8 )
    {
      *(_BYTE *)(v5 + 8) |= 4u;
      v6 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v5 + 24));
      *(_QWORD *)v5 = v6 | *(_QWORD *)v5 & 7LL;
      goto LABEL_29;
    }
    v10 = 0;
    if ( off_4CF6DB8 )
    {
      v11 = v44;
      if ( !v44 )
        goto LABEL_32;
      goto LABEL_12;
    }
LABEL_3:
    ++v4;
    v2 = (_QWORD *)a1[33];
  }
LABEL_14:
  v15 = v2[4];
  v16 = v2[5];
  if ( v16 != v15 )
  {
    v17 = v44;
    v18 = v42;
    v19 = v44 - 1;
    do
    {
      v20 = 0;
      v21 = *(_QWORD *)(*(_QWORD *)v15 + 104LL);
      for ( i = *(_QWORD *)v15 + 96LL; i != v21; v21 = *(_QWORD *)(v21 + 8) )
      {
        if ( v17 )
        {
          v23 = v19 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v24 = (__int64 *)(v18 + 16LL * v23);
          v25 = *v24;
          if ( *v24 == v21 )
          {
LABEL_19:
            v26 = v24[1];
            if ( v26 )
              v20 = v26;
          }
          else
          {
            v27 = 1;
            while ( v25 != -8 )
            {
              v28 = v27 + 1;
              v23 = v19 & (v27 + v23);
              v24 = (__int64 *)(v18 + 16LL * v23);
              v25 = *v24;
              if ( *v24 == v21 )
                goto LABEL_19;
              v27 = v28;
            }
          }
        }
        *(_QWORD *)(v21 + 32) = v20;
      }
      v15 += 8;
    }
    while ( v16 != v15 );
  }
  sub_38D42B0(a1);
  j___libc_free_0(v42);
}
