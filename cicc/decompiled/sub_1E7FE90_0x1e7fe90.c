// Function: sub_1E7FE90
// Address: 0x1e7fe90
//
__int64 __fastcall sub_1E7FE90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // r12
  unsigned __int64 v7; // r13
  __int64 v8; // r14
  _BYTE *v9; // rdi
  int v10; // ecx
  __int64 v11; // r15
  int v12; // r12d
  __int64 v13; // r14
  unsigned __int16 *v14; // rdx
  __int64 v15; // kr00_8
  __int16 v16; // ax
  __int64 v17; // rax
  _WORD *v18; // rdx
  __int64 v19; // rdi
  unsigned __int16 *v20; // rax
  unsigned __int16 *i; // r8
  __int64 v22; // rcx
  __int64 v23; // rdx
  unsigned int v24; // eax
  int v25; // r10d
  __int64 v26; // rdi
  int v27; // ecx
  __int64 v28; // [rsp+18h] [rbp-D8h]
  __int64 v29; // [rsp+20h] [rbp-D0h]
  __int64 v30; // [rsp+28h] [rbp-C8h]
  _BYTE *v31; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v32; // [rsp+38h] [rbp-B8h]
  _BYTE s[176]; // [rsp+40h] [rbp-B0h] BYREF

  v30 = a2;
  v5 = *(_QWORD *)(a1 + 552) + 8LL * *(int *)(a2 + 48);
  if ( *(_DWORD *)v5 == -1 )
  {
    *(_BYTE *)(v5 + 4) = 0;
    v7 = *(unsigned int *)(a1 + 320);
    v8 = a1;
    v9 = s;
    v32 = 0x2000000000LL;
    v31 = s;
    if ( (unsigned int)v7 > 0x20 )
    {
      a2 = (__int64)s;
      sub_16CD150((__int64)&v31, s, v7, 4, a5, (int)&v31);
      v9 = v31;
    }
    LODWORD(v32) = v7;
    if ( 4 * v7 )
    {
      a2 = 0;
      memset(v9, 0, 4 * v7);
    }
    v10 = 0;
    v11 = v8 + 272;
    if ( v30 + 24 != *(_QWORD *)(v30 + 32) )
    {
      v29 = v5;
      v12 = 0;
      v28 = v8;
      v13 = *(_QWORD *)(v30 + 32);
      do
      {
        v14 = *(unsigned __int16 **)(v13 + 16);
        v15 = a2;
        a2 = *v14;
        switch ( *v14 )
        {
          case 0u:
          case 8u:
          case 0xAu:
          case 0xEu:
          case 0xFu:
          case 0x2Du:
            break;
          default:
            a2 = v15;
            switch ( *v14 )
            {
              case 2u:
              case 3u:
              case 4u:
              case 6u:
              case 9u:
              case 0xCu:
              case 0xDu:
              case 0x11u:
              case 0x12u:
                goto LABEL_20;
              default:
                v16 = *(_WORD *)(v13 + 46);
                ++v12;
                if ( (v16 & 4) != 0 || (v16 & 8) == 0 )
                {
                  v17 = (*((_QWORD *)v14 + 1) >> 4) & 1LL;
                }
                else
                {
                  a2 = 16;
                  LOBYTE(v17) = sub_1E15D00(v13, 0x10u, 1);
                }
                if ( (_BYTE)v17 )
                  *(_BYTE *)(v29 + 4) = 1;
                if ( (unsigned __int8)sub_1F4B670(v11) )
                {
                  a2 = v13;
                  v18 = (_WORD *)sub_1F4B8B0(v11, v13);
                  if ( (*v18 & 0x3FFF) != 0x3FFF )
                  {
                    v19 = (unsigned __int16)v18[1];
                    a2 = *(_QWORD *)(*(_QWORD *)(v28 + 448) + 136LL);
                    v20 = (unsigned __int16 *)(a2 + 4 * v19);
                    for ( i = (unsigned __int16 *)(a2 + 4 * (v19 + (unsigned __int16)v18[2]));
                          v20 != i;
                          *(_DWORD *)&v31[4 * v22] += a2 )
                    {
                      v22 = *v20;
                      v20 += 2;
                      a2 = *(v20 - 1);
                    }
                  }
                }
                break;
            }
            break;
        }
LABEL_20:
        if ( (*(_BYTE *)v13 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v13 + 46) & 8) != 0 )
            v13 = *(_QWORD *)(v13 + 8);
        }
        v13 = *(_QWORD *)(v13 + 8);
      }
      while ( v30 + 24 != v13 );
      v10 = v12;
      v8 = v28;
      v5 = v29;
    }
    *(_DWORD *)v5 = v10;
    v23 = 0;
    v24 = v7 * *(_DWORD *)(v30 + 48);
    v25 = v24 + v7;
    if ( (_DWORD)v7 )
    {
      do
      {
        v26 = v24++;
        v27 = *(_DWORD *)(*(_QWORD *)(v8 + 464) + v23) * *(_DWORD *)&v31[v23];
        v23 += 4;
        *(_DWORD *)(*(_QWORD *)(v8 + 600) + 4 * v26) = v27;
      }
      while ( v24 != v25 );
    }
    if ( v31 != s )
      _libc_free((unsigned __int64)v31);
  }
  return v5;
}
