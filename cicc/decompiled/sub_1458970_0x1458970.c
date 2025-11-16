// Function: sub_1458970
// Address: 0x1458970
//
__int64 __fastcall sub_1458970(__int64 a1)
{
  char v1; // dl
  _BYTE *v2; // rsi
  int v3; // eax
  _BYTE *v4; // rdi
  _BYTE *v5; // rcx
  __int64 v6; // rbx
  _QWORD *v7; // r8
  _QWORD *v8; // rbx
  _QWORD *v9; // r15
  char v10; // dl
  __int64 v11; // r14
  _QWORD *v12; // rax
  _QWORD *v13; // rsi
  _QWORD *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rbx
  _QWORD *v17; // rax
  char v18; // dl
  unsigned int v19; // r12d
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rax
  __int64 v23; // r14
  char v24; // dl
  char v25; // dl
  _QWORD *v26; // rdi
  _QWORD *v27; // rcx
  __int64 v28; // rax
  _QWORD *v29; // r8
  _QWORD *v30; // rdx
  _QWORD *v31; // rsi
  __int64 v32; // rax
  _QWORD *v33; // rsi
  unsigned int v34; // [rsp+14h] [rbp-FCh] BYREF
  __int64 v35; // [rsp+18h] [rbp-F8h] BYREF
  unsigned int *v36; // [rsp+20h] [rbp-F0h]
  _BYTE *v37; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v38; // [rsp+30h] [rbp-E0h]
  _BYTE v39[64]; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v40; // [rsp+78h] [rbp-98h] BYREF
  _BYTE *v41; // [rsp+80h] [rbp-90h]
  _BYTE *v42; // [rsp+88h] [rbp-88h]
  __int64 v43; // [rsp+90h] [rbp-80h]
  int v44; // [rsp+98h] [rbp-78h]
  _BYTE v45[112]; // [rsp+A0h] [rbp-70h] BYREF

  v36 = &v34;
  v37 = v39;
  v38 = 0x800000000LL;
  v35 = a1;
  v34 = 0;
  v40 = 0;
  v41 = v45;
  v42 = v45;
  v43 = 8;
  v44 = 0;
  sub_1412190((__int64)&v40, a1);
  if ( v1 )
  {
    ++*v36;
    sub_1458920((__int64)&v37, &v35);
  }
  v2 = v37;
  v3 = v38;
  v4 = v37;
LABEL_4:
  v5 = &v2[8 * v3];
  if ( v3 )
  {
    while ( 2 )
    {
      v6 = *((_QWORD *)v5 - 1);
      LODWORD(v38) = --v3;
      switch ( *(_WORD *)(v6 + 24) )
      {
        case 0:
        case 0xA:
          v5 -= 8;
          if ( v3 )
            continue;
          goto LABEL_26;
        case 1:
        case 2:
        case 3:
          v16 = *(_QWORD *)(v6 + 32);
          v17 = v41;
          if ( v42 != v41 )
            goto LABEL_23;
          v26 = &v41[8 * HIDWORD(v43)];
          if ( v41 == (_BYTE *)v26 )
            goto LABEL_71;
          v27 = 0;
          do
          {
            if ( v16 == *v17 )
              goto LABEL_24;
            if ( *v17 == -2 )
              v27 = v17;
            ++v17;
          }
          while ( v26 != v17 );
          if ( !v27 )
          {
LABEL_71:
            if ( HIDWORD(v43) < (unsigned int)v43 )
            {
              ++HIDWORD(v43);
              *v26 = v16;
              ++v40;
              goto LABEL_44;
            }
LABEL_23:
            sub_16CCBA0(&v40, v16);
            v2 = v37;
            if ( v18 )
              goto LABEL_44;
            goto LABEL_24;
          }
          goto LABEL_43;
        case 4:
        case 5:
        case 7:
        case 8:
        case 9:
          v7 = *(_QWORD **)(v6 + 32);
          v8 = &v7[*(_QWORD *)(v6 + 40)];
          if ( v7 == v8 )
            goto LABEL_4;
          v9 = v7;
          break;
        case 6:
          v21 = (unsigned __int64)v42;
          v22 = v41;
          v23 = *(_QWORD *)(v6 + 32);
          if ( v42 != v41 )
            goto LABEL_32;
          v29 = &v42[8 * HIDWORD(v43)];
          if ( v42 == (_BYTE *)v29 )
            goto LABEL_69;
          v30 = v42;
          v31 = 0;
          do
          {
            if ( v23 == *v30 )
              goto LABEL_33;
            if ( *v30 == -2 )
              v31 = v30;
            ++v30;
          }
          while ( v29 != v30 );
          if ( v31 )
          {
            *v31 = v23;
            --v44;
            ++v40;
          }
          else
          {
LABEL_69:
            if ( HIDWORD(v43) >= (unsigned int)v43 )
            {
LABEL_32:
              sub_16CCBA0(&v40, *(_QWORD *)(v6 + 32));
              v21 = (unsigned __int64)v42;
              v22 = v41;
              if ( !v24 )
              {
LABEL_33:
                v16 = *(_QWORD *)(v6 + 40);
                if ( (_QWORD *)v21 != v22 )
                  goto LABEL_34;
                goto LABEL_60;
              }
            }
            else
            {
              ++HIDWORD(v43);
              *v29 = v23;
              ++v40;
            }
          }
          ++*v36;
          v32 = (unsigned int)v38;
          if ( (unsigned int)v38 >= HIDWORD(v38) )
          {
            sub_16CD150(&v37, v39, 0, 8);
            v32 = (unsigned int)v38;
          }
          *(_QWORD *)&v37[8 * v32] = v23;
          v22 = v41;
          LODWORD(v38) = v38 + 1;
          v16 = *(_QWORD *)(v6 + 40);
          if ( v42 != v41 )
            goto LABEL_34;
LABEL_60:
          v33 = &v22[HIDWORD(v43)];
          if ( v33 != v22 )
          {
            v27 = 0;
            while ( v16 != *v22 )
            {
              if ( *v22 == -2 )
                v27 = v22;
              if ( v33 == ++v22 )
              {
                if ( v27 )
                {
LABEL_43:
                  *v27 = v16;
                  --v44;
                  ++v40;
                  goto LABEL_44;
                }
                goto LABEL_67;
              }
            }
            goto LABEL_35;
          }
LABEL_67:
          if ( HIDWORD(v43) < (unsigned int)v43 )
          {
            ++HIDWORD(v43);
            *v33 = v16;
            ++v40;
LABEL_44:
            ++*v36;
            v28 = (unsigned int)v38;
            if ( (unsigned int)v38 >= HIDWORD(v38) )
            {
              sub_16CD150(&v37, v39, 0, 8);
              v28 = (unsigned int)v38;
            }
            *(_QWORD *)&v37[8 * v28] = v16;
            v2 = v37;
            v3 = v38 + 1;
            LODWORD(v38) = v38 + 1;
            v4 = v37;
            goto LABEL_4;
          }
LABEL_34:
          sub_16CCBA0(&v40, v16);
          if ( v25 )
            goto LABEL_44;
LABEL_35:
          v2 = v37;
LABEL_24:
          v3 = v38;
          v4 = v2;
          goto LABEL_4;
      }
      break;
    }
    while ( 1 )
    {
      v11 = *v9;
      v12 = v41;
      if ( v42 == v41 )
      {
        v13 = &v41[8 * HIDWORD(v43)];
        if ( v41 != (_BYTE *)v13 )
        {
          v14 = 0;
          while ( v11 != *v12 )
          {
            if ( *v12 == -2 )
              v14 = v12;
            if ( v13 == ++v12 )
            {
              if ( !v14 )
                goto LABEL_47;
              *v14 = v11;
              --v44;
              ++v40;
              goto LABEL_19;
            }
          }
          goto LABEL_9;
        }
LABEL_47:
        if ( HIDWORD(v43) < (unsigned int)v43 )
          break;
      }
      sub_16CCBA0(&v40, *v9);
      if ( v10 )
        goto LABEL_19;
LABEL_9:
      if ( v8 == ++v9 )
        goto LABEL_35;
    }
    ++HIDWORD(v43);
    *v13 = v11;
    ++v40;
LABEL_19:
    ++*v36;
    v15 = (unsigned int)v38;
    if ( (unsigned int)v38 >= HIDWORD(v38) )
    {
      sub_16CD150(&v37, v39, 0, 8);
      v15 = (unsigned int)v38;
    }
    *(_QWORD *)&v37[8 * v15] = v11;
    LODWORD(v38) = v38 + 1;
    goto LABEL_9;
  }
LABEL_26:
  v19 = v34;
  if ( v42 != v41 )
  {
    _libc_free((unsigned __int64)v42);
    v4 = v37;
  }
  if ( v4 != v39 )
    _libc_free((unsigned __int64)v4);
  return v19;
}
