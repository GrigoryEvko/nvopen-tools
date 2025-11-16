// Function: sub_145A6A0
// Address: 0x145a6a0
//
void __fastcall sub_145A6A0(__int64 a1, _BYTE **a2)
{
  char v2; // dl
  _BYTE *v3; // rsi
  int v4; // eax
  _BYTE *v5; // rdi
  _BYTE *v6; // rcx
  __int64 v7; // rbx
  _QWORD *v8; // r8
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  char v11; // dl
  __int64 v12; // rax
  __int64 v13; // r14
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  _QWORD *v16; // rdx
  __int64 v17; // rbx
  _QWORD *v18; // rax
  char v19; // dl
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rax
  __int64 v23; // r14
  char v24; // dl
  __int64 v25; // rax
  _QWORD *v26; // rsi
  _QWORD *v27; // rcx
  _QWORD *v28; // r8
  _QWORD *v29; // rdx
  _QWORD *v30; // rsi
  _QWORD *v31; // rdi
  __int64 v32; // [rsp+18h] [rbp-F8h] BYREF
  _BYTE **v33; // [rsp+20h] [rbp-F0h]
  _BYTE *v34; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v35; // [rsp+30h] [rbp-E0h]
  _BYTE v36[64]; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v37; // [rsp+78h] [rbp-98h] BYREF
  _BYTE *v38; // [rsp+80h] [rbp-90h]
  _BYTE *v39; // [rsp+88h] [rbp-88h]
  __int64 v40; // [rsp+90h] [rbp-80h]
  int v41; // [rsp+98h] [rbp-78h]
  _BYTE v42[112]; // [rsp+A0h] [rbp-70h] BYREF

  v34 = v36;
  v33 = a2;
  v35 = 0x800000000LL;
  v32 = a1;
  v37 = 0;
  v38 = v42;
  v39 = v42;
  v40 = 8;
  v41 = 0;
  sub_1412190((__int64)&v37, a1);
  if ( v2 )
  {
    if ( *(_WORD *)(v32 + 24) == 7 )
      **v33 = 1;
    else
      sub_1458920((__int64)&v34, &v32);
  }
  v3 = v34;
  v4 = v35;
  v5 = v34;
LABEL_5:
  while ( 1 )
  {
    v6 = &v3[8 * v4];
    if ( !v4 )
      break;
    while ( 2 )
    {
      v7 = *((_QWORD *)v6 - 1);
      LODWORD(v35) = --v4;
      switch ( *(_WORD *)(v7 + 24) )
      {
        case 0:
        case 0xA:
          v6 -= 8;
          if ( v4 )
            continue;
          goto LABEL_34;
        case 1:
        case 2:
        case 3:
          v17 = *(_QWORD *)(v7 + 32);
          v18 = v38;
          if ( v39 != v38 )
            goto LABEL_28;
          v31 = &v38[8 * HIDWORD(v40)];
          if ( v38 != (_BYTE *)v31 )
          {
            v27 = 0;
            do
            {
              if ( v17 == *v18 )
                goto LABEL_26;
              if ( *v18 == -2 )
                v27 = v18;
              ++v18;
            }
            while ( v31 != v18 );
            if ( v27 )
              goto LABEL_71;
          }
          if ( HIDWORD(v40) >= (unsigned int)v40 )
            goto LABEL_28;
          ++HIDWORD(v40);
          *v31 = v17;
          ++v37;
          goto LABEL_29;
        case 4:
        case 5:
        case 7:
        case 8:
        case 9:
          v8 = *(_QWORD **)(v7 + 32);
          v9 = &v8[*(_QWORD *)(v7 + 40)];
          if ( v8 == v9 )
            goto LABEL_5;
          v10 = v8;
          while ( 1 )
          {
            v13 = *v10;
            v14 = v38;
            if ( v39 != v38 )
              goto LABEL_9;
            v15 = &v38[8 * HIDWORD(v40)];
            if ( v38 != (_BYTE *)v15 )
            {
              v16 = 0;
              while ( v13 != *v14 )
              {
                if ( *v14 == -2 )
                  v16 = v14;
                if ( v15 == ++v14 )
                {
                  if ( !v16 )
                    goto LABEL_53;
                  *v16 = v13;
                  --v41;
                  ++v37;
                  if ( *(_WORD *)(v13 + 24) != 7 )
                    goto LABEL_11;
                  goto LABEL_24;
                }
              }
LABEL_14:
              if ( v9 == ++v10 )
                goto LABEL_25;
              continue;
            }
LABEL_53:
            if ( HIDWORD(v40) < (unsigned int)v40 )
            {
              ++HIDWORD(v40);
              *v15 = v13;
              ++v37;
            }
            else
            {
LABEL_9:
              sub_16CCBA0(&v37, *v10);
              if ( !v11 )
                goto LABEL_14;
            }
            if ( *(_WORD *)(v13 + 24) != 7 )
            {
LABEL_11:
              v12 = (unsigned int)v35;
              if ( (unsigned int)v35 >= HIDWORD(v35) )
              {
                sub_16CD150(&v34, v36, 0, 8);
                v12 = (unsigned int)v35;
              }
              *(_QWORD *)&v34[8 * v12] = v13;
              LODWORD(v35) = v35 + 1;
              goto LABEL_14;
            }
LABEL_24:
            ++v10;
            **v33 = 1;
            if ( v9 == v10 )
              goto LABEL_25;
          }
        case 6:
          v21 = (unsigned __int64)v39;
          v22 = v38;
          v23 = *(_QWORD *)(v7 + 32);
          if ( v39 != v38 )
            goto LABEL_40;
          v28 = &v39[8 * HIDWORD(v40)];
          if ( v39 == (_BYTE *)v28 )
            goto LABEL_78;
          v29 = v39;
          v30 = 0;
          break;
      }
      break;
    }
    do
    {
      if ( v23 == *v29 )
        goto LABEL_45;
      if ( *v29 == -2 )
        v30 = v29;
      ++v29;
    }
    while ( v28 != v29 );
    if ( v30 )
    {
      *v30 = v23;
      --v41;
      ++v37;
      if ( *(_WORD *)(v23 + 24) == 7 )
        goto LABEL_63;
    }
    else
    {
LABEL_78:
      if ( HIDWORD(v40) >= (unsigned int)v40 )
      {
LABEL_40:
        sub_16CCBA0(&v37, *(_QWORD *)(v7 + 32));
        v21 = (unsigned __int64)v39;
        v22 = v38;
        if ( !v24 )
          goto LABEL_45;
      }
      else
      {
        ++HIDWORD(v40);
        *v28 = v23;
        ++v37;
      }
      if ( *(_WORD *)(v23 + 24) == 7 )
      {
LABEL_63:
        **v33 = 1;
        v21 = (unsigned __int64)v39;
        v22 = v38;
        goto LABEL_45;
      }
    }
    v25 = (unsigned int)v35;
    if ( (unsigned int)v35 >= HIDWORD(v35) )
    {
      sub_16CD150(&v34, v36, 0, 8);
      v25 = (unsigned int)v35;
    }
    *(_QWORD *)&v34[8 * v25] = v23;
    v21 = (unsigned __int64)v39;
    LODWORD(v35) = v35 + 1;
    v22 = v38;
LABEL_45:
    v17 = *(_QWORD *)(v7 + 40);
    if ( (_QWORD *)v21 != v22 )
      goto LABEL_28;
    v26 = &v22[HIDWORD(v40)];
    if ( v22 == v26 )
    {
LABEL_74:
      if ( HIDWORD(v40) < (unsigned int)v40 )
      {
        ++HIDWORD(v40);
        *v26 = v17;
        ++v37;
        goto LABEL_29;
      }
LABEL_28:
      sub_16CCBA0(&v37, v17);
      if ( !v19 )
        goto LABEL_25;
LABEL_29:
      if ( *(_WORD *)(v17 + 24) == 7 )
      {
LABEL_72:
        **v33 = 1;
        goto LABEL_25;
      }
LABEL_30:
      v20 = (unsigned int)v35;
      if ( (unsigned int)v35 >= HIDWORD(v35) )
      {
        sub_16CD150(&v34, v36, 0, 8);
        v20 = (unsigned int)v35;
      }
      *(_QWORD *)&v34[8 * v20] = v17;
      v3 = v34;
      v4 = v35 + 1;
      LODWORD(v35) = v35 + 1;
      v5 = v34;
    }
    else
    {
      v27 = 0;
      while ( v17 != *v22 )
      {
        if ( *v22 == -2 )
          v27 = v22;
        if ( v26 == ++v22 )
        {
          if ( !v27 )
            goto LABEL_74;
LABEL_71:
          *v27 = v17;
          --v41;
          ++v37;
          if ( *(_WORD *)(v17 + 24) != 7 )
            goto LABEL_30;
          goto LABEL_72;
        }
      }
LABEL_25:
      v3 = v34;
LABEL_26:
      v4 = v35;
      v5 = v3;
    }
  }
LABEL_34:
  if ( v39 != v38 )
  {
    _libc_free((unsigned __int64)v39);
    v5 = v34;
  }
  if ( v5 != v36 )
    _libc_free((unsigned __int64)v5);
}
