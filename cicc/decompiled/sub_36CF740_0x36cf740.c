// Function: sub_36CF740
// Address: 0x36cf740
//
__int64 __fastcall sub_36CF740(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r14
  __int64 v5; // rdx
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r12
  __int64 v8; // rbx
  unsigned __int64 v9; // r15
  unsigned __int64 *v10; // rbx
  unsigned __int64 v11; // r12
  unsigned __int64 *v12; // rbx
  __int64 v14; // rbx
  __int64 v15; // r12
  unsigned __int64 v16; // r13
  __int64 v17; // rdx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // r15
  unsigned __int64 *v22; // r14
  unsigned __int64 v23; // r12
  unsigned __int64 *v24; // rbx
  unsigned int v25; // eax
  __int64 v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rax
  _QWORD *v31; // rdi
  __int64 v32; // rdx
  _QWORD *v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // rdi
  __int64 v36; // rdx
  _QWORD *v37; // rax
  __int64 v38; // [rsp+0h] [rbp-350h]
  __int64 v39; // [rsp+8h] [rbp-348h]
  unsigned int v40; // [rsp+8h] [rbp-348h]
  unsigned __int64 v41; // [rsp+10h] [rbp-340h] BYREF
  unsigned int v42; // [rsp+18h] [rbp-338h]
  _BYTE v43[816]; // [rsp+20h] [rbp-330h] BYREF

  v2 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v2 != 25 )
  {
    if ( *(_BYTE *)a2 == 85
      && !*(_BYTE *)v2
      && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a2 + 80)
      && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
    {
      v25 = *(_DWORD *)(v2 + 36);
      if ( v25 == 9279 )
      {
        v36 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        v37 = *(_QWORD **)(v36 + 24);
        if ( *(_DWORD *)(v36 + 32) > 0x40u )
          v37 = (_QWORD *)*v37;
        if ( ((unsigned __int8)v37 & 1) == 0 )
          return 0;
      }
      else if ( v25 > 0x243F )
      {
        if ( v25 == 9284 )
        {
          v32 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
          v33 = *(_QWORD **)(v32 + 24);
          if ( *(_DWORD *)(v32 + 32) > 0x40u )
            v33 = (_QWORD *)*v33;
          if ( ((unsigned __int8)v33 & 1) != 0 )
            return 0;
        }
        else if ( v25 == 9376 )
        {
          v30 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
          v31 = *(_QWORD **)(v30 + 24);
          if ( *(_DWORD *)(v30 + 32) > 0x40u )
            v31 = (_QWORD *)*v31;
          if ( sub_CEA480((int)v31) )
            return 0;
        }
      }
      else
      {
        switch ( v25 )
        {
          case 0x22EAu:
            v34 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
            v35 = *(_QWORD **)(v34 + 24);
            if ( *(_DWORD *)(v34 + 32) > 0x40u )
              v35 = (_QWORD *)*v35;
            if ( (sub_CE1180((__int64)v35) & 0x1E0000000000LL) == 0 )
              return 1;
            break;
          case 0x230Fu:
            v26 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
            v27 = *(_QWORD **)(v26 + 24);
            if ( *(_DWORD *)(v26 + 32) > 0x40u )
              v27 = (_QWORD *)*v27;
            if ( ((unsigned __int16)v27 & 0x1E0) == 0xE0 )
              return 0;
            break;
          case 0x22E9u:
            v28 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
            v29 = *(_QWORD **)(v28 + 24);
            if ( *(_DWORD *)(v28 + 32) > 0x40u )
              v29 = (_QWORD *)*v29;
            if ( (unsigned __int8)sub_CE1160((__int64)v29) == 6 )
              return 0;
            break;
        }
      }
    }
    return 255;
  }
  if ( *(_BYTE *)(v2 + 96) )
    return 255;
  sub_B428A0((__int64 *)&v41, *(_BYTE **)(v2 + 56), *(_QWORD *)(v2 + 64));
  v3 = v41;
  v4 = v41 + 192LL * v42;
  if ( v4 == v41 )
  {
LABEL_29:
    if ( (_BYTE *)v3 != v43 )
      _libc_free(v3);
    return 0;
  }
  while ( !*(_BYTE *)(v3 + 10) )
  {
    if ( (*(_BYTE *)v3 & 2) != 0 )
    {
      v14 = *(_QWORD *)(v3 + 16);
      v15 = v14 + 32LL * *(unsigned int *)(v3 + 24);
      if ( v14 != v15 )
      {
        while ( sub_2241AC0(v14, "{memory}") )
        {
          v14 += 32;
          if ( v15 == v14 )
            goto LABEL_6;
        }
        break;
      }
    }
LABEL_6:
    v3 += 192LL;
    if ( v4 == v3 )
    {
      v39 = v41;
      v3 = v41 + 192LL * v42;
      if ( v41 != v3 )
      {
        do
        {
          v5 = *(unsigned int *)(v3 - 120);
          v6 = *(_QWORD *)(v3 - 128);
          v3 -= 192LL;
          v7 = v6 + 56 * v5;
          if ( v6 != v7 )
          {
            do
            {
              v8 = *(unsigned int *)(v7 - 40);
              v9 = *(_QWORD *)(v7 - 48);
              v7 -= 56LL;
              v10 = (unsigned __int64 *)(v9 + 32 * v8);
              if ( (unsigned __int64 *)v9 != v10 )
              {
                do
                {
                  v10 -= 4;
                  if ( (unsigned __int64 *)*v10 != v10 + 2 )
                    j_j___libc_free_0(*v10);
                }
                while ( (unsigned __int64 *)v9 != v10 );
                v9 = *(_QWORD *)(v7 + 8);
              }
              if ( v9 != v7 + 24 )
                _libc_free(v9);
            }
            while ( v6 != v7 );
            v6 = *(_QWORD *)(v3 + 64);
          }
          if ( v6 != v3 + 80 )
            _libc_free(v6);
          v11 = *(_QWORD *)(v3 + 16);
          v12 = (unsigned __int64 *)(v11 + 32LL * *(unsigned int *)(v3 + 24));
          if ( (unsigned __int64 *)v11 != v12 )
          {
            do
            {
              v12 -= 4;
              if ( (unsigned __int64 *)*v12 != v12 + 2 )
                j_j___libc_free_0(*v12);
            }
            while ( (unsigned __int64 *)v11 != v12 );
            v11 = *(_QWORD *)(v3 + 16);
          }
          if ( v11 != v3 + 32 )
            _libc_free(v11);
        }
        while ( v39 != v3 );
        v3 = v41;
      }
      goto LABEL_29;
    }
  }
  v40 = 255;
  v38 = v41;
  v16 = v41 + 192LL * v42;
  if ( v41 != v16 )
  {
    do
    {
      v17 = *(unsigned int *)(v16 - 120);
      v18 = *(_QWORD *)(v16 - 128);
      v16 -= 192LL;
      v19 = v18 + 56 * v17;
      if ( v18 != v19 )
      {
        do
        {
          v20 = *(unsigned int *)(v19 - 40);
          v21 = *(_QWORD *)(v19 - 48);
          v19 -= 56LL;
          v20 *= 32;
          v22 = (unsigned __int64 *)(v21 + v20);
          if ( v21 != v21 + v20 )
          {
            do
            {
              v22 -= 4;
              if ( (unsigned __int64 *)*v22 != v22 + 2 )
                j_j___libc_free_0(*v22);
            }
            while ( (unsigned __int64 *)v21 != v22 );
            v21 = *(_QWORD *)(v19 + 8);
          }
          if ( v21 != v19 + 24 )
            _libc_free(v21);
        }
        while ( v18 != v19 );
        v18 = *(_QWORD *)(v16 + 64);
      }
      if ( v18 != v16 + 80 )
        _libc_free(v18);
      v23 = *(_QWORD *)(v16 + 16);
      v24 = (unsigned __int64 *)(v23 + 32LL * *(unsigned int *)(v16 + 24));
      if ( (unsigned __int64 *)v23 != v24 )
      {
        do
        {
          v24 -= 4;
          if ( (unsigned __int64 *)*v24 != v24 + 2 )
            j_j___libc_free_0(*v24);
        }
        while ( (unsigned __int64 *)v23 != v24 );
        v23 = *(_QWORD *)(v16 + 16);
      }
      if ( v23 != v16 + 32 )
        _libc_free(v23);
    }
    while ( v38 != v16 );
    v16 = v41;
  }
  if ( (_BYTE *)v16 != v43 )
    _libc_free(v16);
  return v40;
}
