// Function: sub_2CCDB60
// Address: 0x2ccdb60
//
void __fastcall sub_2CCDB60(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned int a5,
        char a6,
        char a7,
        _QWORD *a8,
        __int64 a9)
{
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rax
  int v12; // r15d
  unsigned int v13; // ebx
  int v15; // eax
  unsigned int v16; // r14d
  int v17; // esi
  __int64 *v18; // r14
  __int64 **v19; // r15
  __int64 **v20; // r9
  __int64 v21; // rcx
  int v22; // [rsp+4h] [rbp-4Ch]

  v9 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v9 + 8) != 14 )
    BUG();
  v10 = *(_DWORD *)(v9 + 8);
  v11 = *(_QWORD *)(a3 + 8);
  v12 = v10 >> 8;
  if ( *(_BYTE *)(v11 + 8) != 14 )
    BUG();
  v13 = 1;
  v15 = *(_DWORD *)(v11 + 8) >> 8;
  if ( a5 )
    v13 = a5;
  v22 = v15;
  v16 = (v13 | 4) & -(v13 | 4);
  if ( !(unsigned __int8)sub_CE9220(a9) && (_BYTE)qword_50139A8 && v12 == 101 )
    v12 = 5;
  if ( v16 == 1 )
  {
    v17 = v13;
    v18 = (__int64 *)sub_BCB2B0(a8);
  }
  else
  {
    v17 = v13 / v16;
    if ( v16 == 2 )
      v18 = (__int64 *)sub_BCB2C0(a8);
    else
      v18 = (__int64 *)sub_BCB2D0(a8);
  }
  if ( v17 != 1 )
  {
    if ( v17 == 2 )
      v18 = sub_BCD420(v18, 2);
    else
      v18 = (__int64 *)sub_BCDA70(v18, v17);
  }
  v19 = (__int64 **)sub_BCE760((__int64 **)v18, v12);
  v20 = (__int64 **)sub_BCE760((__int64 **)v18, v22);
  if ( *(_BYTE *)a1 == 85
    && (v21 = *(_QWORD *)(a1 - 32)) != 0
    && !*(_BYTE *)v21
    && *(_QWORD *)(v21 + 24) == *(_QWORD *)(a1 + 80)
    && (*(_BYTE *)(v21 + 33) & 0x20) != 0
    && *(_DWORD *)(v21 + 36) == 241 )
  {
    sub_2CCB3B0(a1, (__int64)v18, a2, v19, a3, v20, a4, v13, a6, a7, (__int64)a8, a9);
  }
  else
  {
    sub_2CCCE20(a1, (__int64)v18, a2, v19, a3, v20, a4, v13, a6, a7, (__int64)a8, a9);
  }
}
