// Function: sub_2047BD0
// Address: 0x2047bd0
//
__int64 __fastcall sub_2047BD0(_BYTE *a1)
{
  unsigned int v1; // r12d
  unsigned __int8 v2; // r13
  __int64 v3; // rdx
  __int64 v5; // r9
  unsigned int v6; // r15d
  unsigned int v7; // edx
  unsigned int v8; // r14d
  __int64 *v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rbx
  char v12; // dl
  __int64 *v13; // rsi
  __int64 *v14; // rdx
  __int64 *v15; // r8
  __int64 v16; // rbx
  int v17; // r8d
  int v18; // r9d
  _QWORD *v19; // r12
  unsigned __int8 v20; // al
  __int64 v21; // rax
  unsigned int v22; // r12d
  char v23; // dl
  __int64 v24; // rax
  unsigned int v25; // r13d
  unsigned int v26; // edx
  __int64 v27; // rdi
  _QWORD *v28; // r8
  unsigned __int8 v29; // al
  _QWORD *v30; // rcx
  int v31; // r9d
  _QWORD *v32; // r8
  _QWORD *v33; // rax
  _QWORD *v34; // [rsp+8h] [rbp-1A8h]
  _QWORD *v35; // [rsp+10h] [rbp-1A0h]
  unsigned int v36; // [rsp+20h] [rbp-190h]
  _QWORD *v37; // [rsp+20h] [rbp-190h]
  unsigned __int8 v38; // [rsp+3Bh] [rbp-175h]
  unsigned int v39; // [rsp+3Ch] [rbp-174h]
  _QWORD *v40; // [rsp+40h] [rbp-170h] BYREF
  __int64 v41; // [rsp+48h] [rbp-168h]
  _QWORD v42[16]; // [rsp+50h] [rbp-160h] BYREF
  __int64 v43; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 *v44; // [rsp+D8h] [rbp-D8h]
  __int64 *v45; // [rsp+E0h] [rbp-D0h]
  __int64 v46; // [rsp+E8h] [rbp-C8h]
  int v47; // [rsp+F0h] [rbp-C0h]
  _BYTE v48[184]; // [rsp+F8h] [rbp-B8h] BYREF

  v1 = 0;
  v2 = a1[16];
  if ( v2 <= 0x17u )
    return v1;
  v3 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
    return v1;
  switch ( v2 )
  {
    case '#':
    case '\'':
    case '2':
    case '3':
    case '4':
      goto LABEL_8;
    case '$':
    case '(':
      LOBYTE(v1) = v2 == 76 || (unsigned __int8)(*(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL) - 1) <= 5u;
      if ( !(_BYTE)v1 )
        return v1;
      v1 = 0;
      if ( a1[17] >> 1 != 127 )
        return v1;
LABEL_8:
      v5 = *(_QWORD *)(v3 + 32);
      v1 = 0;
      v39 = v5;
      if ( !(_DWORD)v5 || ((unsigned int)v5 & ((_DWORD)v5 - 1)) != 0 )
        return v1;
      v6 = *(_QWORD *)(v3 + 32);
      v43 = 0;
      v7 = 1;
      v8 = 0;
      v40 = v42;
      v41 = 0x1000000001LL;
      v9 = (__int64 *)v48;
      v44 = (__int64 *)v48;
      v45 = (__int64 *)v48;
      v46 = 16;
      v47 = 0;
      v42[0] = a1;
      v10 = (__int64 *)v48;
      break;
    default:
      return 0;
  }
  while ( 2 )
  {
    if ( !v7 )
    {
LABEL_15:
      v1 = v8;
      goto LABEL_16;
    }
    while ( 1 )
    {
      v11 = v40[v7 - 1];
      LODWORD(v41) = v7 - 1;
      if ( v9 == v10 )
      {
        v13 = &v9[HIDWORD(v46)];
        if ( v9 != v13 )
        {
          v14 = v9;
          v15 = 0;
          while ( v11 != *v14 )
          {
            if ( *v14 == -2 )
            {
              v15 = v14;
              if ( v13 == v14 + 1 )
                goto LABEL_26;
              ++v14;
            }
            else if ( v13 == ++v14 )
            {
              if ( !v15 )
                goto LABEL_61;
LABEL_26:
              *v15 = v11;
              --v47;
              ++v43;
              goto LABEL_27;
            }
          }
          goto LABEL_14;
        }
LABEL_61:
        if ( HIDWORD(v46) < (unsigned int)v46 )
          break;
      }
      sub_16CCBA0((__int64)&v43, v11);
      v10 = v45;
      v9 = v44;
      if ( v12 )
        goto LABEL_27;
LABEL_14:
      v7 = v41;
      if ( !(_DWORD)v41 )
        goto LABEL_15;
    }
    ++HIDWORD(v46);
    *v13 = v11;
    ++v43;
LABEL_27:
    v16 = *(_QWORD *)(v11 + 8);
    if ( !v16 )
    {
LABEL_42:
      v7 = v41;
      v10 = v45;
      v9 = v44;
      continue;
    }
    break;
  }
  v38 = v2;
  while ( 1 )
  {
    v19 = sub_1648700(v16);
    v20 = *((_BYTE *)v19 + 16);
    if ( v20 <= 0x17u )
      break;
    if ( v20 == 77 || v38 == v20 )
    {
      v23 = *(_BYTE *)(*v19 + 8LL);
      if ( v23 == 16 )
        v23 = *(_BYTE *)(**(_QWORD **)(*v19 + 16LL) + 8LL);
      if ( ((unsigned __int8)(v23 - 1) <= 5u || v20 == 76) && v20 != 77 && *((_BYTE *)v19 + 17) >> 1 != 127 )
        break;
      v24 = (unsigned int)v41;
      if ( (unsigned int)v41 >= HIDWORD(v41) )
      {
        sub_16CD150((__int64)&v40, v42, 0, 8, v17, v18);
        v24 = (unsigned int)v41;
      }
      v40[v24] = v19;
      LODWORD(v41) = v41 + 1;
    }
    else if ( v20 == 85 )
    {
      if ( v6 == 1
        || v39 > *(_DWORD *)(*v19 + 32LL)
        || *(_BYTE *)(*(_QWORD *)(sub_13CF970((__int64)v19) + 24) + 16LL) != 9 )
      {
        break;
      }
      v6 >>= 1;
      v25 = 0;
      do
      {
        if ( (unsigned int)sub_15FA9D0(*(v19 - 3), v25) != v25 + v6 )
          goto LABEL_60;
        ++v25;
      }
      while ( v25 != v6 );
      v26 = v6;
      if ( v39 > v6 )
      {
        while ( 1 )
        {
          v36 = v26;
          if ( (unsigned int)sub_15FA9D0(*(v19 - 3), v26) != -1 )
            break;
          v26 = v36 + 1;
          if ( v36 + 1 == v39 )
            goto LABEL_68;
        }
LABEL_60:
        v1 = 0;
        v10 = v45;
        v9 = v44;
        goto LABEL_16;
      }
LABEL_68:
      v27 = v19[1];
      if ( !v27 )
        goto LABEL_60;
      if ( *(_QWORD *)(v27 + 8) )
        goto LABEL_60;
      v28 = sub_1648700(v27);
      v29 = *((_BYTE *)v28 + 16);
      if ( v38 != v29 || v29 <= 0x17u )
        goto LABEL_60;
      v34 = v28;
      v35 = (_QWORD *)sub_13CF970((__int64)v28);
      v37 = (_QWORD *)*v35;
      v30 = *(_QWORD **)sub_13CF970((__int64)v19);
      v32 = v34;
      v33 = (_QWORD *)v35[3];
      if ( (v30 != v37 || v19 != v33) && (v19 != v37 || v30 != v33) )
        goto LABEL_60;
      if ( (unsigned int)v41 >= HIDWORD(v41) )
      {
        sub_16CD150((__int64)&v40, v42, 0, 8, (int)v34, v31);
        v32 = v34;
      }
      v40[(unsigned int)v41] = v32;
      LODWORD(v41) = v41 + 1;
    }
    else
    {
      if ( v20 != 83 )
        break;
      if ( v6 != 1 )
        break;
      v21 = *(_QWORD *)(sub_13CF970((__int64)v19) + 24);
      if ( *(_BYTE *)(v21 + 16) != 13 )
        break;
      v22 = *(_DWORD *)(v21 + 32);
      LOBYTE(v8) = v22 <= 0x40 ? *(_QWORD *)(v21 + 24) == 0 : v22 == (unsigned int)sub_16A57B0(v21 + 24);
      if ( !(_BYTE)v8 )
        break;
      v6 = 1;
    }
    v16 = *(_QWORD *)(v16 + 8);
    if ( !v16 )
    {
      v2 = v38;
      goto LABEL_42;
    }
  }
  v10 = v45;
  v9 = v44;
  v1 = 0;
LABEL_16:
  if ( v10 != v9 )
    _libc_free((unsigned __int64)v10);
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  return v1;
}
