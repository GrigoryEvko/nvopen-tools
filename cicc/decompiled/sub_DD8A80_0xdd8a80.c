// Function: sub_DD8A80
// Address: 0xdd8a80
//
__int64 __fastcall sub_DD8A80(__int64 a1, unsigned int a2, _BYTE *a3, _BYTE *a4, _BYTE *a5, _BYTE *a6)
{
  unsigned int v8; // ebx
  unsigned int v10; // eax
  _BYTE *v11; // rax
  __int64 v12; // rsi
  __int64 *v13; // rax
  _BYTE *v14; // rcx
  __int64 *v15; // r14
  char v16; // al
  __int64 *v17; // rdx
  __int64 v18; // rsi
  _BYTE *v19; // [rsp+8h] [rbp-28h]

  if ( a4 == a6 )
  {
    v10 = sub_B52F50(a2);
    a6 = a5;
    a4 = a3;
    v8 = v10;
  }
  else
  {
    v8 = a2;
    if ( a3 != a5 )
      return 0;
  }
  if ( *((_WORD *)a6 + 12) != 15 )
    return 0;
  v11 = (_BYTE *)*((_QWORD *)a6 - 1);
  if ( *v11 != 55 )
    return 0;
  v12 = *((_QWORD *)v11 - 8);
  v19 = a4;
  if ( !v12 || !*((_QWORD *)v11 - 4) )
    return 0;
  v13 = sub_DD8400(a1, v12);
  v14 = v19;
  v15 = v13;
  if ( v8 - 36 <= 1 )
  {
    v17 = v13;
    v18 = 37;
  }
  else
  {
    if ( v8 - 40 > 1 )
      return 0;
    v16 = sub_DBED40(a1, (__int64)v13);
    v14 = v19;
    if ( !v16 )
      return 0;
    v17 = v15;
    v18 = 41;
  }
  return sub_DC3A60(a1, v18, v17, v14);
}
