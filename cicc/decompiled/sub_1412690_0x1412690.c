// Function: sub_1412690
// Address: 0x1412690
//
__int64 __fastcall sub_1412690(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // r14
  unsigned __int64 v6; // r14
  __int64 v7; // rdx
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r12
  __int64 v10; // rbx
  unsigned __int64 v11; // r15
  _QWORD *v12; // rbx
  unsigned __int64 v13; // r12
  _QWORD *v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // r13
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  unsigned __int64 v22; // r15
  _QWORD *v23; // r14
  unsigned __int64 v24; // r14
  _QWORD *v25; // rbx
  unsigned __int64 v26; // [rsp+0h] [rbp-350h]
  __int64 v27; // [rsp+8h] [rbp-348h]
  unsigned __int64 v28; // [rsp+8h] [rbp-348h]
  unsigned __int64 v29; // [rsp+8h] [rbp-348h]
  unsigned __int64 v30; // [rsp+10h] [rbp-340h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-338h]
  _BYTE v32[816]; // [rsp+20h] [rbp-330h] BYREF

  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
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
    case 0x38:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4B:
    case 0x4C:
    case 0x4D:
    case 0x4F:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
    case 0x58:
      return 0;
    case 0x1D:
    case 0x21:
    case 0x35:
    case 0x36:
    case 0x37:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x4A:
    case 0x50:
    case 0x51:
    case 0x52:
      return 1;
    case 0x4E:
      v15 = *(_QWORD *)(a1 - 24);
      v1 = 1;
      if ( *(_BYTE *)(v15 + 16) != 20 )
        return v1;
      v1 = *(unsigned __int8 *)(v15 + 96);
      if ( (_BYTE)v1 )
        return v1;
      sub_15F1410(&v30, *(_QWORD *)(v15 + 56), *(_QWORD *)(v15 + 64));
      v16 = v30;
      v17 = v30 + 192LL * v31;
      if ( v30 == v17 )
        goto LABEL_61;
      break;
  }
  do
  {
    if ( *(_BYTE *)(v16 + 10) )
      goto LABEL_8;
    if ( (*(_BYTE *)v16 & 2) != 0 )
    {
      v3 = *(_QWORD *)(v16 + 16);
      v4 = 32LL * *(unsigned int *)(v16 + 24);
      v27 = v3 + v4;
      if ( v3 + v4 != v3 )
      {
        v5 = *(_QWORD *)(v16 + 16);
        while ( (unsigned int)sub_2241AC0(v5, "{memory}") )
        {
          v5 += 32;
          if ( v27 == v5 )
            goto LABEL_38;
        }
LABEL_8:
        v28 = v30;
        v6 = v30 + 192LL * v31;
        if ( v30 != v6 )
        {
          do
          {
            v7 = *(unsigned int *)(v6 - 120);
            v8 = *(_QWORD *)(v6 - 128);
            v6 -= 192LL;
            v9 = v8 + 56 * v7;
            if ( v8 != v9 )
            {
              do
              {
                v10 = *(unsigned int *)(v9 - 40);
                v11 = *(_QWORD *)(v9 - 48);
                v9 -= 56LL;
                v12 = (_QWORD *)(v11 + 32 * v10);
                if ( (_QWORD *)v11 != v12 )
                {
                  do
                  {
                    v12 -= 4;
                    if ( (_QWORD *)*v12 != v12 + 2 )
                      j_j___libc_free_0(*v12, v12[2] + 1LL);
                  }
                  while ( (_QWORD *)v11 != v12 );
                  v11 = *(_QWORD *)(v9 + 8);
                }
                if ( v11 != v9 + 24 )
                  _libc_free(v11);
              }
              while ( v8 != v9 );
              v8 = *(_QWORD *)(v6 + 64);
            }
            if ( v8 != v6 + 80 )
              _libc_free(v8);
            v13 = *(_QWORD *)(v6 + 16);
            v14 = (_QWORD *)(v13 + 32LL * *(unsigned int *)(v6 + 24));
            if ( (_QWORD *)v13 != v14 )
            {
              do
              {
                v14 -= 4;
                if ( (_QWORD *)*v14 != v14 + 2 )
                  j_j___libc_free_0(*v14, v14[2] + 1LL);
              }
              while ( (_QWORD *)v13 != v14 );
              v13 = *(_QWORD *)(v6 + 16);
            }
            if ( v13 != v6 + 32 )
              _libc_free(v13);
          }
          while ( v28 != v6 );
          v6 = v30;
        }
        if ( (_BYTE *)v6 != v32 )
          _libc_free(v6);
        return 1;
      }
    }
LABEL_38:
    v16 += 192LL;
  }
  while ( v17 != v16 );
  v26 = v30;
  v17 = v30 + 192LL * v31;
  if ( v30 != v17 )
  {
    do
    {
      v18 = *(unsigned int *)(v17 - 120);
      v19 = *(_QWORD *)(v17 - 128);
      v17 -= 192LL;
      v29 = v19;
      v20 = v19 + 56 * v18;
      if ( v19 != v20 )
      {
        do
        {
          v21 = *(unsigned int *)(v20 - 40);
          v22 = *(_QWORD *)(v20 - 48);
          v20 -= 56LL;
          v21 *= 32;
          v23 = (_QWORD *)(v22 + v21);
          if ( v22 != v22 + v21 )
          {
            do
            {
              v23 -= 4;
              if ( (_QWORD *)*v23 != v23 + 2 )
                j_j___libc_free_0(*v23, v23[2] + 1LL);
            }
            while ( (_QWORD *)v22 != v23 );
            v22 = *(_QWORD *)(v20 + 8);
          }
          if ( v22 != v20 + 24 )
            _libc_free(v22);
        }
        while ( v29 != v20 );
        v29 = *(_QWORD *)(v17 + 64);
      }
      if ( v29 != v17 + 80 )
        _libc_free(v29);
      v24 = *(_QWORD *)(v17 + 16);
      v25 = (_QWORD *)(v24 + 32LL * *(unsigned int *)(v17 + 24));
      if ( (_QWORD *)v24 != v25 )
      {
        do
        {
          v25 -= 4;
          if ( (_QWORD *)*v25 != v25 + 2 )
            j_j___libc_free_0(*v25, v25[2] + 1LL);
        }
        while ( (_QWORD *)v24 != v25 );
        v24 = *(_QWORD *)(v17 + 16);
      }
      if ( v24 != v17 + 32 )
        _libc_free(v24);
    }
    while ( v26 != v17 );
    v17 = v30;
  }
LABEL_61:
  if ( (_BYTE *)v17 != v32 )
    _libc_free(v17);
  return v1;
}
