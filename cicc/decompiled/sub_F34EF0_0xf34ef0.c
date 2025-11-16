// Function: sub_F34EF0
// Address: 0xf34ef0
//
__int64 __fastcall sub_F34EF0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // r15
  unsigned __int64 v4; // rdi
  int v5; // eax
  _BYTE *v6; // rbx
  _BYTE *v7; // rdi
  unsigned int j; // r13d
  __int64 i; // [rsp+8h] [rbp-78h]
  unsigned int v11; // [rsp+1Ch] [rbp-64h]
  _BYTE v12[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v13; // [rsp+40h] [rbp-40h]

  v2 = 0;
  v3 = *(_QWORD *)(a1 + 80);
  for ( i = a1 + 72; i != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    while ( 1 )
    {
      if ( !v3 )
        BUG();
      v4 = *(_QWORD *)(v3 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 == v3 + 24 )
      {
        v6 = 0;
      }
      else
      {
        if ( !v4 )
          BUG();
        v5 = *(unsigned __int8 *)(v4 - 24);
        v6 = 0;
        v7 = (_BYTE *)(v4 - 24);
        if ( (unsigned int)(v5 - 30) < 0xB )
          v6 = v7;
      }
      v11 = sub_B46E30((__int64)v6);
      if ( v11 > 1 && *v6 != 33 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( i == v3 )
        return v2;
    }
    for ( j = 0; j != v11; ++j )
    {
      v13 = 257;
      v2 -= (sub_F451F0(v6, j, a2, v12) == 0) - 1;
    }
  }
  return v2;
}
