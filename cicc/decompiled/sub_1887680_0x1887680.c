// Function: sub_1887680
// Address: 0x1887680
//
void __fastcall sub_1887680(__int64 a1, __int64 a2, char a3, double a4, double a5, double a6)
{
  __int64 **v6; // rax
  __int64 *v9; // r14
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 *v14; // r13
  __int64 v15; // rax
  __int64 *v16; // r9
  __int64 *v17; // r13
  __int64 *v18; // r14
  __int64 v19; // rdi
  unsigned int v20; // esi
  _QWORD *v21; // rdi
  int v22; // esi
  unsigned int v23; // edx
  __int64 v24; // r8
  int v25; // r13d
  __int64 *v26; // r10
  unsigned int v27; // edx
  unsigned int v28; // edi
  unsigned int v29; // r8d
  __int64 v30; // rax
  __int64 v31; // [rsp+20h] [rbp-A0h] BYREF
  __int64 *v32; // [rsp+28h] [rbp-98h] BYREF
  __int64 v33; // [rsp+30h] [rbp-90h] BYREF
  __int64 v34; // [rsp+38h] [rbp-88h]
  _QWORD *v35; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v36; // [rsp+48h] [rbp-78h]
  __int64 *v37; // [rsp+60h] [rbp-60h] BYREF
  __int64 v38; // [rsp+68h] [rbp-58h]
  _BYTE v39[80]; // [rsp+70h] [rbp-50h] BYREF

  v6 = &v35;
  v33 = 0;
  v34 = 1;
  do
    *v6++ = (__int64 *)-8LL;
  while ( v6 != &v37 );
  v9 = *(__int64 **)(a1 + 8);
  v37 = (__int64 *)v39;
  v38 = 0x400000000LL;
  if ( !v9 )
    goto LABEL_23;
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = v9;
        v9 = (__int64 *)v9[1];
        v15 = (__int64)sub_1648700((__int64)v14);
        v12 = *(unsigned __int8 *)(v15 + 16);
        if ( (_BYTE)v12 == 4 )
          goto LABEL_10;
        if ( (_BYTE)v12 == 78 )
          break;
        if ( (unsigned __int8)v12 > 0x10u )
          goto LABEL_15;
        v31 = v15;
        if ( (unsigned __int8)v12 <= 3u )
          goto LABEL_15;
        v12 = v34 & 1;
        if ( (v34 & 1) != 0 )
        {
          v21 = &v35;
          v22 = 3;
        }
        else
        {
          v20 = v36;
          v21 = v35;
          if ( !v36 )
          {
            v27 = v34;
            ++v33;
            v26 = 0;
            v28 = ((unsigned int)v34 >> 1) + 1;
            goto LABEL_51;
          }
          v22 = v36 - 1;
        }
        v23 = v22 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v16 = &v21[v23];
        v24 = *v16;
        if ( v15 != *v16 )
        {
          v25 = 1;
          v26 = 0;
          while ( v24 != -8 )
          {
            if ( v26 || v24 != -16 )
              v16 = v26;
            v23 = v22 & (v25 + v23);
            v24 = v21[v23];
            if ( v15 == v24 )
              goto LABEL_10;
            ++v25;
            v26 = v16;
            v16 = &v21[v23];
          }
          v27 = v34;
          if ( !v26 )
            v26 = v16;
          ++v33;
          v28 = ((unsigned int)v34 >> 1) + 1;
          if ( (_BYTE)v12 )
          {
            v29 = 12;
            v20 = 4;
            goto LABEL_40;
          }
          v20 = v36;
LABEL_51:
          v29 = 3 * v20;
LABEL_40:
          if ( v29 <= 4 * v28 )
          {
            v20 *= 2;
          }
          else if ( v20 - HIDWORD(v34) - v28 > v20 >> 3 )
          {
LABEL_42:
            LODWORD(v34) = (2 * (v27 >> 1) + 2) | v27 & 1;
            if ( *v26 != -8 )
              --HIDWORD(v34);
            *v26 = v15;
            v30 = (unsigned int)v38;
            if ( (unsigned int)v38 >= HIDWORD(v38) )
            {
              sub_16CD150((__int64)&v37, v39, 0, 8, v29, (int)v16);
              v30 = (unsigned int)v38;
            }
            v12 = v31;
            v37[v30] = v31;
            LODWORD(v38) = v38 + 1;
            goto LABEL_10;
          }
          sub_18872D0((__int64)&v33, v20);
          sub_1882E00((__int64)&v33, &v31, &v32);
          v26 = v32;
          v15 = v31;
          v27 = v34;
          goto LABEL_42;
        }
LABEL_10:
        if ( !v9 )
          goto LABEL_18;
      }
      if ( v14 == (__int64 *)((v15 & 0xFFFFFFFFFFFFFFF8LL) - 24) && ((*(_BYTE *)(a1 + 33) & 0x40) != 0 || !a3) )
        goto LABEL_10;
LABEL_15:
      if ( *v14 )
        break;
      *v14 = a2;
      if ( a2 )
        goto LABEL_7;
      if ( !v9 )
        goto LABEL_18;
    }
    v11 = v14[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v11 = v9;
    if ( !v9 )
      break;
    v12 = v9[2] & 3;
    v9[2] = v12 | v11;
    *v14 = a2;
    if ( a2 )
      goto LABEL_7;
  }
  *v14 = a2;
  if ( a2 )
  {
LABEL_7:
    v13 = *(_QWORD *)(a2 + 8);
    v14[1] = v13;
    if ( v13 )
    {
      v12 = (unsigned __int64)(v14 + 1) | *(_QWORD *)(v13 + 16) & 3LL;
      *(_QWORD *)(v13 + 16) = v12;
    }
    v14[2] = (a2 + 8) | v14[2] & 3;
    *(_QWORD *)(a2 + 8) = v14;
    goto LABEL_10;
  }
LABEL_18:
  v17 = v37;
  v18 = &v37[(unsigned int)v38];
  if ( v37 != v18 )
  {
    do
    {
      v19 = *v17++;
      sub_15A5060(v19, (_BYTE *)a1, a2, (__int64 *)v12, a4, a5, a6);
    }
    while ( v18 != v17 );
    v18 = v37;
  }
  if ( v18 != (__int64 *)v39 )
    _libc_free((unsigned __int64)v18);
LABEL_23:
  if ( (v34 & 1) == 0 )
    j___libc_free_0(v35);
}
