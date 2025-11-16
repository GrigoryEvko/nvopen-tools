// Function: sub_8978E0
// Address: 0x8978e0
//
__int64 *__fastcall sub_8978E0(int a1, int a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  int v8; // r14d
  __int64 *v10; // r13
  _BYTE *v11; // rax
  __int64 v12; // r12
  char v13; // cl
  char v14; // bl
  __int64 v15; // rax
  unsigned int v16; // r12d

  v8 = a3;
  if ( (_DWORD)a3 )
    a5 = 0;
  v10 = sub_897810(2u, a5, a3, 0);
  v11 = sub_724D80(12);
  v10[11] = (__int64)v11;
  v12 = (__int64)v11;
  *((_QWORD *)v11 + 16) = a6;
  sub_7249B0((__int64)v11, 0);
  v13 = 4 * (a4 & 1);
  *(_DWORD *)(v12 + 184) = a2;
  v14 = *(_BYTE *)(v12 + 177);
  *(_DWORD *)(v12 + 188) = a1;
  *(_BYTE *)(v12 + 177) = v13 | v14 & 0xFB;
  sub_877D80(v12, v10);
  if ( dword_4F07590 )
  {
    v15 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *(_BYTE *)(v15 + 4) == 8 )
    {
      if ( *(_QWORD *)(v15 + 184) )
      {
        sub_72EE40(v12, 2u, *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 184));
        sub_733310(v12, 0);
      }
    }
  }
  if ( v8 )
    sub_877D70(v12);
  v16 = dword_4F04C3C;
  dword_4F04C3C = 1;
  sub_8756F0(3, (__int64)v10, v10 + 6, 0);
  dword_4F04C3C = v16;
  return v10;
}
