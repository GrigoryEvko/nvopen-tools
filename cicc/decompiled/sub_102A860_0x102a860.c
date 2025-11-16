// Function: sub_102A860
// Address: 0x102a860
//
__int64 __fastcall sub_102A860(_BYTE *a1)
{
  unsigned int v1; // r13d
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  char *v6; // rsi
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rbx
  _QWORD *v12; // r15
  _QWORD *v13; // rbx
  _QWORD *v14; // r12
  _QWORD *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // rax
  _QWORD *v23; // r15
  _QWORD *v24; // r14
  _QWORD *v25; // r14
  _QWORD *v26; // rbx
  __int64 v27; // [rsp+0h] [rbp-350h]
  __int64 v28; // [rsp+8h] [rbp-348h]
  __int64 v29; // [rsp+8h] [rbp-348h]
  __int64 v30; // [rsp+8h] [rbp-348h]
  __int64 v31; // [rsp+10h] [rbp-340h] BYREF
  unsigned int v32; // [rsp+18h] [rbp-338h]
  _BYTE v33[816]; // [rsp+20h] [rbp-330h] BYREF

  switch ( *a1 )
  {
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x27:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3F:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x4D:
    case 0x4E:
    case 0x4F:
    case 0x50:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x5A:
    case 0x5B:
    case 0x5C:
    case 0x5D:
    case 0x5E:
    case 0x5F:
    case 0x60:
      return 0;
    case 0x22:
    case 0x26:
    case 0x28:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x51:
    case 0x56:
    case 0x57:
    case 0x58:
    case 0x59:
      return 1;
    case 0x55:
      v16 = *((_QWORD *)a1 - 4);
      v1 = 1;
      if ( *(_BYTE *)v16 != 25 )
        return v1;
      v1 = *(unsigned __int8 *)(v16 + 96);
      if ( (_BYTE)v1 )
        return v1;
      v6 = *(char **)(v16 + 56);
      sub_B428A0(&v31, v6, *(_QWORD *)(v16 + 64));
      v17 = v31;
      v18 = 192LL * v32;
      v29 = v31 + v18;
      if ( v31 + v18 == v31 )
        goto LABEL_61;
      break;
    default:
      BUG();
  }
  do
  {
    if ( *(_BYTE *)(v17 + 10) )
      goto LABEL_8;
    if ( (*(_BYTE *)v17 & 2) != 0 )
    {
      v3 = *(_QWORD *)(v17 + 16);
      v4 = v3 + 32LL * *(unsigned int *)(v17 + 24);
      if ( v4 != v3 )
      {
        v5 = *(_QWORD *)(v17 + 16);
        while ( 1 )
        {
          v6 = "{memory}";
          if ( !(unsigned int)sub_2241AC0(v5, "{memory}") )
            break;
          v5 += 32;
          if ( v4 == v5 )
            goto LABEL_38;
        }
LABEL_8:
        v28 = v31;
        v7 = v31 + 192LL * v32;
        if ( v31 != v7 )
        {
          do
          {
            v8 = *(unsigned int *)(v7 - 120);
            v9 = *(_QWORD *)(v7 - 128);
            v7 -= 192;
            v10 = v9 + 56 * v8;
            if ( v9 != v10 )
            {
              do
              {
                v11 = *(unsigned int *)(v10 - 40);
                v12 = *(_QWORD **)(v10 - 48);
                v10 -= 56;
                v13 = &v12[4 * v11];
                if ( v12 != v13 )
                {
                  do
                  {
                    v13 -= 4;
                    if ( (_QWORD *)*v13 != v13 + 2 )
                    {
                      v6 = (char *)(v13[2] + 1LL);
                      j_j___libc_free_0(*v13, v6);
                    }
                  }
                  while ( v12 != v13 );
                  v12 = *(_QWORD **)(v10 + 8);
                }
                if ( v12 != (_QWORD *)(v10 + 24) )
                  _libc_free(v12, v6);
              }
              while ( v9 != v10 );
              v9 = *(_QWORD *)(v7 + 64);
            }
            if ( v9 != v7 + 80 )
              _libc_free(v9, v6);
            v14 = *(_QWORD **)(v7 + 16);
            v15 = &v14[4 * *(unsigned int *)(v7 + 24)];
            if ( v14 != v15 )
            {
              do
              {
                v15 -= 4;
                if ( (_QWORD *)*v15 != v15 + 2 )
                {
                  v6 = (char *)(v15[2] + 1LL);
                  j_j___libc_free_0(*v15, v6);
                }
              }
              while ( v14 != v15 );
              v14 = *(_QWORD **)(v7 + 16);
            }
            if ( v14 != (_QWORD *)(v7 + 32) )
              _libc_free(v14, v6);
          }
          while ( v28 != v7 );
          v7 = v31;
        }
        if ( (_BYTE *)v7 != v33 )
          _libc_free(v7, v6);
        return 1;
      }
    }
LABEL_38:
    v17 += 192;
  }
  while ( v29 != v17 );
  v27 = v31;
  v17 = v31 + 192LL * v32;
  if ( v31 != v17 )
  {
    do
    {
      v19 = *(unsigned int *)(v17 - 120);
      v20 = *(_QWORD *)(v17 - 128);
      v17 -= 192;
      v30 = v20;
      v21 = v20 + 56 * v19;
      if ( v20 != v21 )
      {
        do
        {
          v22 = *(unsigned int *)(v21 - 40);
          v23 = *(_QWORD **)(v21 - 48);
          v21 -= 56;
          v22 *= 32;
          v24 = (_QWORD *)((char *)v23 + v22);
          if ( v23 != (_QWORD *)((char *)v23 + v22) )
          {
            do
            {
              v24 -= 4;
              if ( (_QWORD *)*v24 != v24 + 2 )
              {
                v6 = (char *)(v24[2] + 1LL);
                j_j___libc_free_0(*v24, v6);
              }
            }
            while ( v23 != v24 );
            v23 = *(_QWORD **)(v21 + 8);
          }
          if ( v23 != (_QWORD *)(v21 + 24) )
            _libc_free(v23, v6);
        }
        while ( v30 != v21 );
        v30 = *(_QWORD *)(v17 + 64);
      }
      if ( v30 != v17 + 80 )
        _libc_free(v30, v6);
      v25 = *(_QWORD **)(v17 + 16);
      v26 = &v25[4 * *(unsigned int *)(v17 + 24)];
      if ( v25 != v26 )
      {
        do
        {
          v26 -= 4;
          if ( (_QWORD *)*v26 != v26 + 2 )
          {
            v6 = (char *)(v26[2] + 1LL);
            j_j___libc_free_0(*v26, v6);
          }
        }
        while ( v25 != v26 );
        v25 = *(_QWORD **)(v17 + 16);
      }
      if ( v25 != (_QWORD *)(v17 + 32) )
        _libc_free(v25, v6);
    }
    while ( v27 != v17 );
    v17 = v31;
  }
LABEL_61:
  if ( (_BYTE *)v17 != v33 )
    _libc_free(v17, v6);
  return v1;
}
