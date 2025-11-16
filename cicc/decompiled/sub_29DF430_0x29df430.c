// Function: sub_29DF430
// Address: 0x29df430
//
__int64 __fastcall sub_29DF430(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  _QWORD *v8; // rbx
  _QWORD *v9; // r13
  _QWORD *k; // r14
  __int64 v12; // rdi
  _QWORD *v13; // rax
  size_t v14; // rdx
  __int64 v15; // rdi
  char v16; // bl
  char *i; // r12
  unsigned int v18; // eax
  char v19; // al
  unsigned __int64 *v20; // r13
  unsigned __int64 *v21; // r12
  _QWORD *v22; // [rsp+10h] [rbp-230h]
  bool v23; // [rsp+37h] [rbp-209h]
  __int64 v24; // [rsp+38h] [rbp-208h]
  __int64 v25; // [rsp+40h] [rbp-200h]
  __int16 v27; // [rsp+6Eh] [rbp-1D2h] BYREF
  __int64 v28; // [rsp+70h] [rbp-1D0h] BYREF
  unsigned int v29; // [rsp+78h] [rbp-1C8h] BYREF
  char v30; // [rsp+7Ch] [rbp-1C4h]
  unsigned int v31; // [rsp+80h] [rbp-1C0h] BYREF
  char v32; // [rsp+84h] [rbp-1BCh]
  unsigned int j; // [rsp+88h] [rbp-1B8h] BYREF
  char v34; // [rsp+8Ch] [rbp-1B4h]
  char *v35; // [rsp+90h] [rbp-1B0h] BYREF
  size_t v36; // [rsp+98h] [rbp-1A8h]
  __int64 v37; // [rsp+A0h] [rbp-1A0h] BYREF
  __int64 v38; // [rsp+A8h] [rbp-198h]
  __int64 v39; // [rsp+B0h] [rbp-190h]
  __int64 v40; // [rsp+B8h] [rbp-188h]
  _QWORD *v41; // [rsp+C0h] [rbp-180h]
  __int64 v42; // [rsp+C8h] [rbp-178h]
  _QWORD v43[6]; // [rsp+D0h] [rbp-170h] BYREF
  unsigned __int64 *v44; // [rsp+100h] [rbp-140h] BYREF
  __int64 v45; // [rsp+108h] [rbp-138h]
  _BYTE v46[304]; // [rsp+110h] [rbp-130h] BYREF

  v4 = a1;
  v6 = a3 + 72;
  v7 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v8 = *(_QWORD **)(a3 + 80);
  v25 = v7;
  v24 = v7 + 8;
  if ( (_QWORD *)v6 == v8 )
  {
    v9 = 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    while ( 1 )
    {
      v9 = (_QWORD *)v8[4];
      if ( v9 != v8 + 3 )
        break;
      v8 = (_QWORD *)v8[1];
      if ( (_QWORD *)v6 == v8 )
        goto LABEL_7;
      if ( !v8 )
        BUG();
    }
  }
  if ( (_QWORD *)v6 != v8 )
  {
    k = v9;
    do
    {
      if ( !k )
        BUG();
      if ( *((_BYTE *)k - 24) == 85
        && (!(unsigned __int8)sub_A73ED0(k + 6, 23) && !(unsigned __int8)sub_B49560((__int64)(k - 3), 23)
         || (unsigned __int8)sub_A73ED0(k + 6, 4)
         || (unsigned __int8)sub_B49560((__int64)(k - 3), 4)) )
      {
        v12 = *(k - 7);
        if ( v12 )
        {
          if ( !*(_BYTE *)v12 && *(_QWORD *)(v12 + 24) == k[7] )
          {
            v35 = (char *)sub_BD5D20(v12);
            v36 = v14;
            v23 = sub_97F890(*(_QWORD *)(v25 + 8), v35, v14);
            if ( v23 )
            {
              v44 = (unsigned __int64 *)v46;
              v45 = 0x800000000LL;
              sub_C0B8E0((__int64)(k - 3), (__int64)&v44);
              v28 = sub_B43CA0((__int64)(k - 3));
              v37 = 0;
              v38 = 0;
              v39 = 0;
              v40 = 0;
              v41 = v43;
              v42 = 0;
              sub_29DF010((__int64)&v37, (__int64)v44, (__int64)&v44[4 * (unsigned int)v45]);
              v43[1] = &v35;
              v43[2] = &v37;
              v43[5] = k - 3;
              v43[3] = &v44;
              v43[4] = &v28;
              v43[0] = v24;
              v15 = *(_QWORD *)(v25 + 8);
              v29 = 0;
              v30 = 0;
              v31 = 0;
              v32 = 0;
              sub_980260(v15, v35, v36, (__int64)&v29, (__int64)&v31);
              v27 = 256;
              v22 = v8;
              v16 = 0;
              for ( i = (char *)&v27; ; v16 = *i )
              {
                v34 = 0;
                v18 = 2;
                j = 2;
                while ( v29 >= v18 )
                {
                  sub_29DEB60((__int64)v43, (__int64)&j, v16);
                  v18 = 2 * j;
                  j *= 2;
                  if ( v34 )
                  {
                    if ( !v30 )
                      break;
                  }
                }
                v34 = 1;
                v19 = v23;
                for ( j = 2; (!v19 || v32) && j <= v31; j *= 2 )
                {
                  sub_29DEB60((__int64)v43, (__int64)&j, v16);
                  v19 = v34;
                }
                if ( ++i == (char *)&v28 )
                  break;
              }
              v8 = v22;
              sub_C0A420((__int64)(k - 3), v44, (unsigned int)v45);
              if ( v41 != v43 )
                _libc_free((unsigned __int64)v41);
              sub_C7D6A0(v38, 16LL * (unsigned int)v40, 8);
              v20 = v44;
              v21 = &v44[4 * (unsigned int)v45];
              if ( v44 != v21 )
              {
                do
                {
                  v21 -= 4;
                  if ( (unsigned __int64 *)*v21 != v21 + 2 )
                    j_j___libc_free_0(*v21);
                }
                while ( v20 != v21 );
                v21 = v44;
              }
              if ( v21 != (unsigned __int64 *)v46 )
                _libc_free((unsigned __int64)v21);
            }
          }
        }
      }
      for ( k = (_QWORD *)k[1]; ; k = (_QWORD *)v8[4] )
      {
        v13 = v8 - 3;
        if ( !v8 )
          v13 = 0;
        if ( k != v13 + 6 )
          break;
        v8 = (_QWORD *)v8[1];
        if ( (_QWORD *)v6 == v8 )
          goto LABEL_26;
        if ( !v8 )
          BUG();
      }
    }
    while ( (_QWORD *)v6 != v8 );
LABEL_26:
    v4 = a1;
  }
LABEL_7:
  *(_QWORD *)(v4 + 48) = 0;
  *(_QWORD *)(v4 + 8) = v4 + 32;
  *(_QWORD *)(v4 + 56) = v4 + 80;
  *(_QWORD *)(v4 + 16) = 0x100000002LL;
  *(_QWORD *)(v4 + 64) = 2;
  *(_QWORD *)(v4 + 32) = &qword_4F82400;
  *(_DWORD *)(v4 + 72) = 0;
  *(_BYTE *)(v4 + 76) = 1;
  *(_DWORD *)(v4 + 24) = 0;
  *(_BYTE *)(v4 + 28) = 1;
  *(_QWORD *)v4 = 1;
  return v4;
}
