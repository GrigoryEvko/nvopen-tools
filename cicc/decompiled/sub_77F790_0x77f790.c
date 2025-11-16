// Function: sub_77F790
// Address: 0x77f790
//
_BOOL8 __fastcall sub_77F790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, int a6)
{
  __m128i *v9; // rax
  __int64 v10; // r15
  _BOOL4 v11; // eax
  int v13; // eax
  __m128i *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  _BOOL4 v19; // [rsp+2Ch] [rbp-54h] BYREF
  __m128i v20; // [rsp+30h] [rbp-50h] BYREF
  __m128i v21[4]; // [rsp+40h] [rbp-40h] BYREF

  v9 = sub_7707D0(a1, *(const __m128i **)(a3 + 16));
  *(_QWORD *)(a3 + 16) = v9;
  v10 = (__int64)v9;
  sub_620D80(&v20, a4);
  sub_620DE0(v21, a5);
  sub_621F20(&v20, v21, 1, &v19);
  if ( v19 )
    goto LABEL_2;
  if ( *(_BYTE *)(v10 + 173) != 6 )
  {
    v13 = sub_620E90(v10);
    v14 = (__m128i *)(v10 + 176);
    if ( a6 )
      sub_621670(v14, v13, v20.m128i_i16, 1, &v19);
    else
      sub_621340((unsigned __int16 *)v14, v13, v20.m128i_i16, 1, &v19);
LABEL_9:
    v11 = v19;
    if ( !v19 )
      return !v11;
    goto LABEL_2;
  }
  v15 = sub_77F710(v10, 1);
  v16 = *(_QWORD *)(v15 + 16);
  if ( a6 )
  {
    *(_QWORD *)(v15 + 16) = v16 - a4;
    sub_620D80(v21, *(_QWORD *)(v10 + 192));
    sub_6215F0((unsigned __int16 *)v21, v20.m128i_i16, 1, &v19);
  }
  else
  {
    *(_QWORD *)(v15 + 16) = a4 + v16;
    sub_620D80(v21, *(_QWORD *)(v10 + 192));
    sub_621270((unsigned __int16 *)v21, v20.m128i_i16, 1, &v19);
  }
  if ( !v19 )
  {
    *(_QWORD *)(v10 + 192) = sub_620EE0(v21, 1, &v19);
    goto LABEL_9;
  }
LABEL_2:
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    sub_686CA0(0xA93u, a2, *(_QWORD *)(v10 + 128), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
  }
  v11 = v19;
  return !v11;
}
