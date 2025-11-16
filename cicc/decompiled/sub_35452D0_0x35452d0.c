// Function: sub_35452D0
// Address: 0x35452d0
//
__int16 __fastcall sub_35452D0(__int64 a1, __int64 *a2, int a3)
{
  int v4; // ecx
  int v5; // edx
  __int64 v6; // rsi
  int v7; // ecx
  __int16 result; // ax
  _WORD *v9; // rsi
  __int64 v10; // r13
  char v11; // al
  _WORD *v12; // rax
  int v13; // [rsp+Ch] [rbp-24h]

  if ( *(_BYTE *)(a1 + 40) )
  {
    v4 = *(_DWORD *)(a1 + 480);
    v5 = a3 % v4;
    v6 = *(_QWORD *)(*a2 + 16);
    v7 = v5 + v4;
    if ( v5 < 0 )
      v5 = v7;
    return sub_37F1990(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * v5), v6);
  }
  else
  {
    v9 = (_WORD *)a2[2];
    if ( !v9 )
    {
      v13 = a3;
      v10 = *(_QWORD *)(a1 + 32) + 600LL;
      v11 = sub_2FF7B70(v10);
      a3 = v13;
      if ( v11 )
      {
        v12 = sub_2FF7DB0(v10, *a2);
        a3 = v13;
        a2[2] = (__int64)v12;
        v9 = v12;
      }
      else
      {
        v9 = (_WORD *)a2[2];
      }
    }
    result = *v9 & 0x1FFF;
    if ( result != 0x1FFF )
      return sub_35451F0(a1, v9, a3);
  }
  return result;
}
