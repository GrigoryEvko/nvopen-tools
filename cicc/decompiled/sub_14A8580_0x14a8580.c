// Function: sub_14A8580
// Address: 0x14a8580
//
void __fastcall sub_14A8580(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // r15
  __int64 v8; // r8
  unsigned __int8 v9; // al
  __int64 v10; // rax
  char v11; // al
  _QWORD *v12; // r8
  int v13; // eax
  _QWORD *v14; // rdx
  __int64 v15; // rcx
  _QWORD *v16; // rax
  __int64 v17; // rcx
  unsigned __int64 v18; // r10
  int v19; // esi
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v23; // [rsp+18h] [rbp-98h]
  __int64 v24; // [rsp+20h] [rbp-90h]
  _QWORD *v25; // [rsp+20h] [rbp-90h]
  _BYTE *v26; // [rsp+30h] [rbp-80h] BYREF
  __int64 v27; // [rsp+38h] [rbp-78h]
  _BYTE v28[112]; // [rsp+40h] [rbp-70h] BYREF

  for ( i = *(_QWORD *)(a3 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v8 = sub_1648700(i);
    v9 = *(_BYTE *)(v8 + 16);
    if ( v9 > 0x17u )
    {
      switch ( v9 )
      {
        case 'G':
          sub_14A8580(a1, a2, v8, a4);
          break;
        case '6':
          sub_14A8490(a2, 0, *(_QWORD *)(v8 + 8), a4);
          break;
        case '8':
          v10 = *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
          if ( a3 == v10 )
          {
            if ( v10 )
            {
              v24 = v8;
              v11 = sub_15FA290(v8);
              v12 = (_QWORD *)v24;
              if ( v11 )
              {
                v13 = *(_DWORD *)(v24 + 20);
                v27 = 0x800000000LL;
                v26 = v28;
                v14 = v28;
                v15 = 24 * (1LL - (v13 & 0xFFFFFFF));
                v16 = (_QWORD *)(v24 + v15);
                v17 = -v15;
                v18 = 0xAAAAAAAAAAAAAAABLL * (v17 >> 3);
                v19 = 0;
                if ( (unsigned __int64)v17 > 0xC0 )
                {
                  v22 = v16;
                  v23 = 0xAAAAAAAAAAAAAAABLL * (v17 >> 3);
                  sub_16CD150(&v26, v28, v23, 8);
                  v19 = v27;
                  v16 = v22;
                  v12 = (_QWORD *)v24;
                  LODWORD(v18) = v23;
                  v14 = &v26[8 * (unsigned int)v27];
                }
                if ( v12 != v16 )
                {
                  do
                  {
                    if ( v14 )
                      *v14 = *v16;
                    v16 += 3;
                    ++v14;
                  }
                  while ( v12 != v16 );
                  v19 = v27;
                }
                LODWORD(v27) = v19 + v18;
                v25 = v12;
                v20 = sub_1632FA0(a1);
                v21 = sub_15A9FF0(v20, v25[7], v26, (unsigned int)v27);
                sub_14A8580(a1, a2, v25, v21 + a4);
                if ( v26 != v28 )
                  _libc_free((unsigned __int64)v26);
              }
            }
          }
          break;
      }
    }
  }
}
