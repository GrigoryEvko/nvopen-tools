// Function: sub_2EE8230
// Address: 0x2ee8230
//
__int64 __fastcall sub_2EE8230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 v8; // r14
  _DWORD *v9; // rax
  _BYTE *v11; // rdx
  int v12; // r13d
  _DWORD *i; // rdx
  __int64 v14; // r14
  __int64 v15; // r15
  int v16; // eax
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  int v19; // edx
  char v20; // al
  _WORD *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned __int16 *v24; // rax
  unsigned __int16 *j; // r9
  __int64 v26; // rsi
  int v27; // r10d
  unsigned int v28; // eax
  __int64 v29; // rcx
  __int64 v30; // rdi
  int v32; // [rsp+1Ch] [rbp-D4h]
  __int64 v33; // [rsp+28h] [rbp-C8h]
  _BYTE *v34; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+38h] [rbp-B8h]
  _BYTE v36[176]; // [rsp+40h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a1 + 336) + 8LL * *(int *)(a2 + 24);
  if ( *(_DWORD *)v6 != -1 )
    return v6;
  *(_BYTE *)(v6 + 4) = 0;
  v8 = *(unsigned int *)(a1 + 88);
  v9 = v36;
  v11 = v36;
  v35 = 0x2000000000LL;
  v12 = v8;
  v34 = v36;
  v33 = a2 + 48;
  if ( !v8 )
  {
    v14 = *(_QWORD *)(a2 + 56);
    if ( v14 == v33 )
    {
      *(_DWORD *)v6 = 0;
      goto LABEL_30;
    }
    goto LABEL_11;
  }
  if ( v8 > 0x20 )
  {
    sub_C8D5F0((__int64)&v34, v36, v8, 4u, a5, a6);
    v11 = v34;
    v9 = &v34[4 * (unsigned int)v35];
  }
  for ( i = &v11[4 * v8]; i != v9; ++v9 )
  {
    if ( v9 )
      *v9 = 0;
  }
  LODWORD(v35) = v8;
  v14 = *(_QWORD *)(a2 + 56);
  if ( v14 != v33 )
  {
LABEL_11:
    v32 = 0;
    v15 = 0x800000000000C09LL;
    do
    {
      v16 = *(unsigned __int16 *)(v14 + 68);
      if ( (_WORD)v16 )
      {
        v17 = (unsigned int)(v16 - 9);
        if ( (unsigned __int16)v17 > 0x3Bu || !_bittest64(&v15, v17) )
        {
          v18 = *(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL);
          if ( (v18 & 0x10) == 0 )
          {
            v19 = *(_DWORD *)(v14 + 44);
            ++v32;
            if ( (v19 & 4) != 0 || (v19 & 8) == 0 )
              v20 = (unsigned __int8)v18 >> 7;
            else
              v20 = sub_2E88A90(v14, 128, 1);
            if ( v20 )
              *(_BYTE *)(v6 + 4) = 1;
            if ( (unsigned __int8)sub_2FF7B70(a1 + 40) )
            {
              v21 = (_WORD *)sub_2FF7DB0(a1 + 40, v14);
              if ( (*v21 & 0x1FFF) != 0x1FFF )
              {
                v22 = (unsigned __int16)v21[1];
                v23 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 176LL);
                v24 = (unsigned __int16 *)(v23 + 6 * v22);
                for ( j = (unsigned __int16 *)(v23 + 6 * (v22 + (unsigned __int16)v21[2]));
                      v24 != j;
                      *(_DWORD *)&v34[4 * v26] += *(v24 - 2) )
                {
                  v26 = *v24;
                  v24 += 3;
                }
              }
            }
          }
        }
      }
      if ( (*(_BYTE *)v14 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v14 + 44) & 8) != 0 )
          v14 = *(_QWORD *)(v14 + 8);
      }
      v14 = *(_QWORD *)(v14 + 8);
    }
    while ( v14 != v33 );
    *(_DWORD *)v6 = v32;
    v27 = v12 * *(_DWORD *)(a2 + 24);
    if ( !v12 )
      goto LABEL_30;
    goto LABEL_28;
  }
  *(_DWORD *)v6 = 0;
  v27 = v12 * *(_DWORD *)(a2 + 24);
LABEL_28:
  v28 = 0;
  do
  {
    v29 = v28;
    v30 = v28 + v27;
    ++v28;
    *(_DWORD *)(*(_QWORD *)(a1 + 384) + 4 * v30) = *(_DWORD *)(*(_QWORD *)(a1 + 248) + 4 * v29)
                                                 * *(_DWORD *)&v34[4 * v29];
  }
  while ( v12 != v28 );
LABEL_30:
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
  return v6;
}
