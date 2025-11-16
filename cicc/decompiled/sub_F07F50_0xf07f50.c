// Function: sub_F07F50
// Address: 0xf07f50
//
__int64 __fastcall sub_F07F50(__int64 a1, unsigned __int8 *a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rax
  unsigned __int8 **v8; // rbx
  unsigned __int8 **v9; // r15
  int v10; // r13d
  _BYTE *v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int8 *v14; // r12
  __int64 v15; // rax
  _BYTE *v16; // rsi
  __int64 v17; // r12
  unsigned __int64 v19; // rax
  char v20; // dl
  unsigned __int8 *v21; // rdi
  _BYTE *v24; // [rsp+28h] [rbp-D8h]
  _BYTE *v25; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v26; // [rsp+48h] [rbp-B8h]
  _BYTE v27[48]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v28[8]; // [rsp+80h] [rbp-80h] BYREF
  __int16 v29; // [rsp+C0h] [rbp-40h]

  v6 = a1;
  v25 = v27;
  v26 = 0x600000000LL;
  v7 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v8 = *(unsigned __int8 ***)(a1 - 8);
    v9 = &v8[v7];
  }
  else
  {
    v9 = (unsigned __int8 **)a1;
    v8 = (unsigned __int8 **)(a1 - v7 * 8);
  }
  if ( v8 != v9 )
  {
    v10 = (a3 == 0) + 32;
    do
    {
      v14 = *v8;
      if ( *v8 == a2 )
      {
        if ( a3 )
          v14 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
        else
          v14 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
      }
      else
      {
        v11 = (_BYTE *)*((_QWORD *)a2 - 12);
        LODWORD(v28[0]) = v10;
        BYTE4(v28[0]) = 0;
        if ( *v11 == 82 )
        {
          v24 = v11;
          v19 = sub_B53900((__int64)v11);
          sub_B53630(v19, v28[0]);
          if ( v20 )
          {
            if ( v14 == *((unsigned __int8 **)v24 - 8) )
            {
              v21 = (unsigned __int8 *)*((_QWORD *)v24 - 4);
              if ( v21 )
              {
                if ( sub_98ED60(v21, 0, 0, 0, 0) )
                  v14 = v21;
              }
            }
          }
        }
      }
      v12 = (unsigned int)v26;
      v13 = (unsigned int)v26 + 1LL;
      if ( v13 > HIDWORD(v26) )
      {
        sub_C8D5F0((__int64)&v25, v27, v13, 8u, a5, a6);
        v12 = (unsigned int)v26;
      }
      v8 += 4;
      *(_QWORD *)&v25[8 * v12] = v14;
      LODWORD(v26) = v26 + 1;
    }
    while ( v9 != v8 );
    v6 = a1;
  }
  v15 = sub_B43CC0(v6);
  v16 = v25;
  v28[0] = v15;
  memset(&v28[1], 0, 56);
  v29 = 257;
  v17 = sub_1020E00(v6, v25, (unsigned int)v26, v28);
  if ( v25 != v27 )
    _libc_free(v25, v16);
  return v17;
}
