// Function: sub_1004A20
// Address: 0x1004a20
//
__int64 __fastcall sub_1004A20(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  bool v4; // zf
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // eax
  __int64 result; // rax
  int v9; // r12d
  _QWORD *v10; // [rsp+8h] [rbp-38h] BYREF
  _QWORD *v11; // [rsp+10h] [rbp-30h] BYREF
  __int64 v12; // [rsp+18h] [rbp-28h]

  v10 = 0;
  if ( (unsigned __int8)sub_995B10(&v10, (__int64)a1) )
    return sub_AD62B0(*((_QWORD *)a1 + 1));
  v4 = *a1 == 54;
  v11 = 0;
  v12 = a2;
  if ( v4 && (unsigned __int8)sub_995B10(&v11, *((_QWORD *)a1 - 8)) && *((_QWORD *)a1 - 4) == v12 )
    return sub_AD62B0(*((_QWORD *)a1 + 1));
  if ( !*(_BYTE *)(a3 + 64) )
    goto LABEL_12;
  v5 = *a1;
  if ( (unsigned __int8)v5 <= 0x1Cu )
  {
    if ( (_BYTE)v5 != 5 )
      goto LABEL_12;
    v7 = *((unsigned __int16 *)a1 + 1);
    if ( (*((_WORD *)a1 + 1) & 0xFFFD) != 0xD && (v7 & 0xFFF7) != 0x11 )
      goto LABEL_12;
  }
  else
  {
    if ( (unsigned __int8)v5 > 0x36u )
      goto LABEL_12;
    v6 = 0x40540000000000LL;
    if ( !_bittest64(&v6, v5) )
      goto LABEL_12;
    v7 = (unsigned __int8)v5 - 29;
  }
  if ( v7 != 25 || (a1[1] & 4) == 0 || (result = *((_QWORD *)a1 - 8)) == 0 || a2 != *((_QWORD *)a1 - 4) )
  {
LABEL_12:
    v9 = sub_9AF8B0((__int64)a1, *(_QWORD *)a3, 0, *(_QWORD *)(a3 + 32), *(_QWORD *)(a3 + 40), *(_QWORD *)(a3 + 24), 1);
    v4 = v9 == (unsigned int)sub_BCB060(*((_QWORD *)a1 + 1));
    result = 0;
    if ( v4 )
      return (__int64)a1;
  }
  return result;
}
