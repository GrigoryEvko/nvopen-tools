// Function: sub_34CED20
// Address: 0x34ced20
//
__int64 __fastcall sub_34CED20(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  unsigned __int16 v3; // ax
  __int64 v4; // rdx
  __int64 v5; // r8
  unsigned __int8 v7; // al

  v2 = *(_QWORD *)(a1 + 32);
  v3 = sub_2D5BAE0(v2, *(_QWORD *)(a1 + 16), a2, 0);
  v4 = 1;
  if ( (v3 == 1 || (v5 = 4, v3) && (v4 = v3, *(_QWORD *)(v2 + 8LL * v3 + 112)))
    && (v5 = 1, v7 = *(_BYTE *)(v2 + 500 * v4 + 6510), v7 > 1u) )
  {
    return 3LL * (v7 != 4) + 1;
  }
  else
  {
    return v5;
  }
}
