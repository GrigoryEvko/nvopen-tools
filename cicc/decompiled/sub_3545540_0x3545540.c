// Function: sub_3545540
// Address: 0x3545540
//
__int64 __fastcall sub_3545540(__int64 a1, __int64 *a2, int a3)
{
  int v4; // ecx
  int v5; // edx
  __int64 v6; // rsi
  int v7; // ecx
  _WORD *v9; // r14
  unsigned int v10; // r15d
  __int64 v11; // r14
  _WORD *v12; // rax

  if ( *(_BYTE *)(a1 + 40) )
  {
    v4 = *(_DWORD *)(a1 + 480);
    v5 = a3 % v4;
    v6 = *(_QWORD *)(*a2 + 16);
    v7 = v5 + v4;
    if ( v5 < 0 )
      v5 = v7;
    return sub_37F0C40(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * v5), v6);
  }
  else
  {
    v9 = (_WORD *)a2[2];
    if ( !v9 )
    {
      v11 = *(_QWORD *)(a1 + 32) + 600LL;
      if ( sub_2FF7B70(v11) )
      {
        v12 = sub_2FF7DB0(v11, *a2);
        a2[2] = (__int64)v12;
        v9 = v12;
      }
      else
      {
        v9 = (_WORD *)a2[2];
      }
    }
    v10 = 1;
    if ( (*v9 & 0x1FFF) != 0x1FFF )
    {
      sub_35451F0(a1, v9, a3);
      v10 = sub_3545490(a1) ^ 1;
      sub_35453B0(a1, v9, a3);
    }
    return v10;
  }
}
