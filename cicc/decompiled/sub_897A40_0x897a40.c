// Function: sub_897A40
// Address: 0x897a40
//
_QWORD *__fastcall sub_897A40(int a1, __int64 a2, int a3, int a4, __int64 a5, __int64 a6, const __m128i *a7)
{
  __int64 *v9; // rbx
  _QWORD *v10; // r13
  char v11; // al
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // rax
  unsigned int v15; // r15d
  _QWORD *v16; // r14
  __int64 v18; // rdx

  v9 = sub_897810(3u, a2, a3 == 0, 0);
  v10 = sub_7259C0(14);
  v11 = a4 & 1 | *((_BYTE *)v10 + 161) & 0xFE;
  *((_BYTE *)v10 + 161) = v11;
  *((_BYTE *)v10 + 161) = (2 * (*(_BYTE *)(a6 + 92) & 1)) | v11 & 0xFD;
  v12 = v10[21];
  v13 = *(_DWORD *)(a6 + 168);
  *(_DWORD *)(v12 + 24) = a1;
  *(_DWORD *)(v12 + 28) = v13;
  if ( a5 )
  {
    *(_QWORD *)(v12 + 32) = a5;
    *(_DWORD *)(a6 + 128) = 1;
  }
  sub_8D6090(v10);
  sub_877D80((__int64)v10, v9);
  if ( dword_4F07590 )
  {
    v14 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *(_BYTE *)(v14 + 4) == 8 )
    {
      v18 = *(_QWORD *)(v14 + 184);
      if ( v18 )
      {
        sub_72EE40((__int64)v10, 6u, v18);
        sub_7365B0((__int64)v10, dword_4F04C64);
      }
    }
  }
  if ( a3 )
  {
    v9[11] = (__int64)v10;
LABEL_7:
    v15 = dword_4F04C3C;
    dword_4F04C3C = 1;
    sub_8756F0(3, (__int64)v9, v9 + 6, 0);
    dword_4F04C3C = v15;
    goto LABEL_8;
  }
  sub_877D70((__int64)v10);
  v9[11] = (__int64)v10;
  if ( !a2 || **(_BYTE **)(*(_QWORD *)a2 + 8LL) != 60 )
    goto LABEL_7;
LABEL_8:
  sub_729470((__int64)v10, a7);
  v16 = sub_880AD0((__int64)v9);
  if ( a4 )
  {
    if ( a5 && *(_QWORD *)(a5 + 64) && (unsigned int)sub_8670F0() )
    {
      if ( qword_4F04C18[2] )
      {
        if ( !(unsigned int)sub_866580() )
          *((_BYTE *)v16 + 56) |= 0x10u;
        *((_BYTE *)v9 + 84) |= 0x20u;
        *((_BYTE *)v16 + 56) |= 0x40u;
      }
      else
      {
        *((_BYTE *)v16 + 56) |= 0x20u;
        sub_866580();
        *((_BYTE *)v16 + 56) |= 0x10u;
      }
    }
    else
    {
      sub_866580();
      *((_BYTE *)v16 + 56) |= 0x10u;
    }
    *(_QWORD *)(a6 + 84) = 0x100000001LL;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) |= 1u;
  }
  return v16;
}
