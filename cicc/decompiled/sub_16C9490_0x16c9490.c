// Function: sub_16C9490
// Address: 0x16c9490
//
__int64 __fastcall sub_16C9490(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v7; // rax
  unsigned int v8; // r9d
  __int64 v9; // r13
  unsigned int v10; // r13d
  __int64 v11; // r10
  __int64 v12; // r11
  _BYTE *v13; // rdx
  __int64 *v14; // rax
  __int64 *v16; // r11
  unsigned __int64 v17; // rcx
  int v18; // eax
  unsigned int v19; // r9d
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r13
  __int64 v24; // r9
  __int64 v25; // r8
  __int64 v26; // r12
  _QWORD *v27; // rax
  unsigned int v28; // ecx
  __int64 v29; // r12
  __int64 v31; // [rsp-E0h] [rbp-E0h]
  __int64 v32; // [rsp-E0h] [rbp-E0h]
  unsigned int v33; // [rsp-D8h] [rbp-D8h]
  __int64 v34; // [rsp-D8h] [rbp-D8h]
  __int64 v35; // [rsp-D8h] [rbp-D8h]
  unsigned __int8 v36; // [rsp-D0h] [rbp-D0h]
  _BYTE *v37; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v38; // [rsp-C0h] [rbp-C0h]
  _BYTE v39[16]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v40; // [rsp-A8h] [rbp-A8h] BYREF

  if ( *((_DWORD *)a1 + 2) )
    return 0;
  v4 = a3;
  if ( a4 )
  {
    v7 = *a1;
    v8 = 1;
    v37 = v39;
    v9 = *(_QWORD *)(v7 + 8);
    v38 = 0x800000000LL;
    v10 = v9 + 1;
    if ( v10 )
      v8 = v10;
    v11 = v10;
    v12 = 16LL * v8;
    if ( v10 <= 8 )
    {
      v14 = (__int64 *)v39;
      v13 = v39;
    }
    else
    {
      v31 = 16LL * v8;
      v33 = v8;
      sub_16CD150(&v37, v39, v8, 16);
      v13 = v37;
      v11 = v10;
      v8 = v33;
      v12 = v31;
      v4 = a3;
      v14 = (__int64 *)&v37[16 * (unsigned int)v38];
    }
    v16 = (__int64 *)&v13[v12];
    if ( v16 == v14 )
      goto LABEL_13;
  }
  else
  {
    v11 = 0;
    v10 = 0;
    v37 = v39;
    v16 = &v40;
    v8 = 1;
    v38 = 0x800000000LL;
    v14 = (__int64 *)v39;
  }
  do
  {
    if ( v14 )
    {
      *v14 = 0;
      v14[1] = 0;
    }
    v14 += 2;
  }
  while ( v16 != v14 );
  v13 = v37;
LABEL_13:
  LODWORD(v38) = v8;
  *(_QWORD *)v13 = 0;
  v17 = (unsigned __int64)v37;
  *((_QWORD *)v37 + 1) = v4;
  v18 = sub_16EF020(*a1, a2, v11, v17, 4);
  v19 = 0;
  if ( v18 != 1 )
  {
    if ( v18 )
    {
      *((_DWORD *)a1 + 2) = v18;
    }
    else
    {
      if ( a4 )
      {
        *(_DWORD *)(a4 + 8) = 0;
        if ( v10 )
        {
          v20 = v10 - 1;
          v21 = 0;
          v22 = a4 + 16;
          v23 = 0;
          v24 = 16 * v20;
          while ( 1 )
          {
            v28 = *(_DWORD *)(a4 + 12);
            v29 = *(_QWORD *)&v37[v23];
            if ( v29 == -1 )
            {
              if ( (unsigned int)v21 >= v28 )
              {
                v35 = v24;
                sub_16CD150(a4, v22, 0, 16);
                v21 = *(unsigned int *)(a4 + 8);
                v24 = v35;
              }
              *(_OWORD *)(*(_QWORD *)a4 + 16 * v21) = 0;
              ++*(_DWORD *)(a4 + 8);
              if ( v23 == v24 )
                break;
            }
            else
            {
              v25 = *(_QWORD *)&v37[v23 + 8] - v29;
              v26 = a2 + v29;
              if ( (unsigned int)v21 >= v28 )
              {
                v32 = v24;
                v34 = v25;
                sub_16CD150(a4, v22, 0, 16);
                v21 = *(unsigned int *)(a4 + 8);
                v24 = v32;
                v25 = v34;
              }
              v27 = (_QWORD *)(*(_QWORD *)a4 + 16 * v21);
              *v27 = v26;
              v27[1] = v25;
              ++*(_DWORD *)(a4 + 8);
              if ( v23 == v24 )
                break;
            }
            v21 = *(unsigned int *)(a4 + 8);
            v23 += 16;
          }
        }
      }
      v19 = 1;
    }
  }
  if ( v37 != v39 )
  {
    v36 = v19;
    _libc_free((unsigned __int64)v37);
    return v36;
  }
  return v19;
}
