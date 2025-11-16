// Function: sub_2596E60
// Address: 0x2596e60
//
__int64 __fastcall sub_2596E60(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  int v3; // r13d
  int v4; // edx
  unsigned int v6; // r14d
  __int64 v7; // r13
  __int64 v8; // r15
  int v9; // eax
  _QWORD *v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // r14d
  unsigned __int8 *v14; // rax
  unsigned __int8 *v15; // rax
  unsigned __int8 v16; // [rsp+Fh] [rbp-51h]
  char v17; // [rsp+1Fh] [rbp-41h] BYREF
  __m128i v18[4]; // [rsp+20h] [rbp-40h] BYREF

  v2 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
    v2 = **(_QWORD **)(v2 + 16);
  v3 = *(_DWORD *)(v2 + 8) >> 8;
  if ( **(_DWORD **)a1 != 4 && (v3 != 4 || !(unsigned __int8)sub_CF7060(a2))
    || (v10 = (_QWORD *)sub_B43CA0(*(_QWORD *)(a1 + 8)), !(unsigned __int8)sub_250C0F0(v10)) )
  {
    v4 = *a2;
    if ( (unsigned int)(v4 - 12) > 1 )
    {
      if ( (_BYTE)v4 == 22 )
      {
        v6 = 16;
      }
      else if ( (unsigned __int8)v4 > 3u )
      {
        if ( (_BYTE)v4 == 20 )
        {
          v13 = **(_DWORD **)a1;
          v14 = sub_250CBE0((__int64 *)(*(_QWORD *)(a1 + 16) + 72LL), (__int64)a2);
          if ( !sub_B2F070((__int64)v14, v13) )
            return 1;
          v15 = sub_250CBE0((__int64 *)(*(_QWORD *)(a1 + 16) + 72LL), v13);
          if ( !sub_B2F070((__int64)v15, v3) )
            return 1;
          v4 = *a2;
        }
        v6 = 128;
        if ( (unsigned __int8)v4 > 0x1Cu )
        {
          v6 = 1;
          if ( (_BYTE)v4 != 60 )
          {
            v11 = (unsigned int)(v4 - 34);
            v6 = 128;
            if ( (unsigned __int8)v11 <= 0x33u )
            {
              v12 = 0x8000000000041LL;
              if ( _bittest64(&v12, v11) )
              {
                sub_250D230((unsigned __int64 *)v18, (unsigned __int64)a2, 3, 0);
                v6 = (unsigned __int8)sub_2596DB0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 16), v18, 1, &v17, 0, 0) == 0
                   ? 128
                   : 64;
              }
            }
          }
        }
      }
      else
      {
        if ( (_BYTE)v4 == 3 && (a2[80] & 1) != 0 )
          return 1;
        v6 = (a2[32] & 0xFu) - 7 < 2 ? 4 : 8;
      }
      v7 = *(_QWORD *)(a1 + 8);
      v8 = *(_QWORD *)(a1 + 16);
      if ( v7 )
      {
        v16 = sub_B46420(*(_QWORD *)(a1 + 8));
        v9 = (2 * ((unsigned __int8)sub_B46490(v7) != 0)) | v16;
      }
      else
      {
        v9 = 3;
      }
      sub_2561E50(v8, *(_QWORD *)(a1 + 32), v6, v7, (__int64)a2, *(_QWORD *)(a1 + 40), v9);
    }
  }
  return 1;
}
