// Function: sub_16B5630
// Address: 0x16b5630
//
void __fastcall sub_16B5630(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v5; // r13
  __int64 v6; // r12
  int v7; // r14d
  __int64 v8; // rax
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v23; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE *v24; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+38h] [rbp-B8h]
  _BYTE v26[176]; // [rsp+40h] [rbp-B0h] BYREF

  v5 = a4;
  v24 = v26;
  v25 = 0x8000000000LL;
  if ( !a2 )
  {
LABEL_17:
    if ( a5 )
      goto LABEL_42;
    goto LABEL_18;
  }
  v6 = 0;
  v7 = 0;
  do
  {
    while ( 1 )
    {
      v9 = *(unsigned __int8 *)(a1 + v6);
      if ( !v7 )
      {
        if ( (unsigned __int8)v9 > 0x20u )
        {
          v7 = 2;
          if ( (_BYTE)v9 != 34 )
          {
            if ( (_BYTE)v9 != 92 )
            {
LABEL_23:
              v11 = (unsigned int)v25;
              if ( (unsigned int)v25 >= HIDWORD(v25) )
              {
                sub_16CD150(&v24, v26, 0, 1);
                v11 = (unsigned int)v25;
              }
              v7 = 1;
              v24[v11] = v9;
              LODWORD(v25) = v25 + 1;
              goto LABEL_8;
            }
            v7 = 1;
            v6 = sub_16B0850(a1, a2, v6, (__int64)&v24);
          }
        }
        else
        {
          v8 = 0x100002600LL;
          if ( !_bittest64(&v8, v9) )
            goto LABEL_23;
          if ( (_BYTE)v9 == 10 && a5 )
          {
            v23 = 0;
            sub_16B55E0(a4, &v23);
          }
        }
        goto LABEL_8;
      }
      if ( v7 != 1 )
        break;
      if ( (unsigned __int8)v9 > 0x20u )
      {
        if ( (_BYTE)v9 == 34 )
        {
          v7 = 2;
        }
        else
        {
          if ( (_BYTE)v9 != 92 )
          {
LABEL_35:
            v15 = (unsigned int)v25;
            if ( (unsigned int)v25 >= HIDWORD(v25) )
            {
              sub_16CD150(&v24, v26, 0, 1);
              v15 = (unsigned int)v25;
            }
            v24[v15] = v9;
            LODWORD(v25) = v25 + 1;
            goto LABEL_8;
          }
          v6 = sub_16B0850(a1, a2, v6, (__int64)&v24);
        }
      }
      else
      {
        v12 = 0x100002600LL;
        if ( !_bittest64(&v12, v9) )
          goto LABEL_35;
        v13 = sub_16D3940(a3, v24, (unsigned int)v25);
        v14 = *(unsigned int *)(a4 + 8);
        if ( (unsigned int)v14 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, a4 + 16, 0, 8);
          v14 = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v14) = v13;
        LODWORD(v25) = 0;
        ++*(_DWORD *)(a4 + 8);
        if ( (_BYTE)v9 == 10 && a5 )
        {
          v7 = 0;
          v23 = 0;
          sub_16B55E0(a4, &v23);
        }
        else
        {
          v7 = 0;
        }
      }
LABEL_8:
      if ( ++v6 == a2 )
        goto LABEL_16;
    }
    if ( (_BYTE)v9 == 34 )
    {
      v7 = 1;
      goto LABEL_8;
    }
    if ( (_BYTE)v9 == 92 )
    {
      v7 = 2;
      v6 = sub_16B0850(a1, a2, v6, (__int64)&v24);
      goto LABEL_8;
    }
    v10 = (unsigned int)v25;
    if ( (unsigned int)v25 >= HIDWORD(v25) )
    {
      sub_16CD150(&v24, v26, 0, 1);
      v10 = (unsigned int)v25;
    }
    ++v6;
    v7 = 2;
    v24[v10] = v9;
    LODWORD(v25) = v25 + 1;
  }
  while ( v6 != a2 );
LABEL_16:
  v5 = a4;
  if ( !(_DWORD)v25 )
    goto LABEL_17;
  v16 = sub_16D3940(a3, v24, (unsigned int)v25);
  v17 = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)v17 >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, a4 + 16, 0, 8);
    v17 = *(unsigned int *)(a4 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a4 + 8 * v17) = v16;
  ++*(_DWORD *)(a4 + 8);
  if ( a5 )
  {
LABEL_42:
    v18 = *(unsigned int *)(v5 + 8);
    if ( (unsigned int)v18 >= *(_DWORD *)(v5 + 12) )
    {
      sub_16CD150(v5, v5 + 16, 0, 8);
      v18 = *(unsigned int *)(v5 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v5 + 8 * v18) = 0;
    ++*(_DWORD *)(v5 + 8);
  }
LABEL_18:
  if ( v24 != v26 )
    _libc_free((unsigned __int64)v24);
}
