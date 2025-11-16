// Function: sub_CA8C30
// Address: 0xca8c30
//
char *__fastcall sub_CA8C30(__int64 a1, _QWORD *a2)
{
  char v3; // al
  unsigned __int64 v4; // rsi
  char *v5; // rdi

  v3 = **(_BYTE **)(a1 + 72);
  if ( v3 == 34 )
    return sub_CA8A40(a1, *(char **)(a1 + 72), *(_QWORD *)(a1 + 80), a2);
  v4 = *(_QWORD *)(a1 + 80);
  v5 = *(char **)(a1 + 72);
  if ( v3 == 39 )
    return sub_CA8AE0(v5, v4, a2);
  else
    return sub_CA8B70(v5, v4, a2);
}
