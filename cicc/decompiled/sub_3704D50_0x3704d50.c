// Function: sub_3704D50
// Address: 0x3704d50
//
unsigned __int64 *__fastcall sub_3704D50(unsigned __int64 *a1, _QWORD *a2, __int64 *a3)
{
  __int16 v3; // ax
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v7; // [rsp+8h] [rbp-58h] BYREF
  __int16 v8; // [rsp+10h] [rbp-50h] BYREF
  __int64 v9; // [rsp+12h] [rbp-4Eh]
  __int64 v10; // [rsp+20h] [rbp-40h]
  __int64 v11; // [rsp+28h] [rbp-38h]
  __int64 v12; // [rsp+30h] [rbp-30h]
  __int64 v13; // [rsp+38h] [rbp-28h]
  __int64 v14; // [rsp+40h] [rbp-20h]
  __int64 v15; // [rsp+48h] [rbp-18h]

  v3 = 0;
  if ( a2[1] > 3u )
    v3 = *(_WORD *)(*a2 + 2LL);
  v8 = v3;
  v4 = *a3;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  (*(void (__fastcall **)(__int64 *, __int64 *, _QWORD *, __int16 *))(v4 + 136))(&v7, a3, a2, &v8);
  v5 = v7 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v7 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v7 = 0;
    *a1 = v5 | 1;
    sub_9C66B0(&v7);
  }
  else
  {
    v7 = 0;
    sub_9C66B0(&v7);
    *a1 = 1;
  }
  return a1;
}
