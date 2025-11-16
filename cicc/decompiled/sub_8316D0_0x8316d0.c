// Function: sub_8316D0
// Address: 0x8316d0
//
void __fastcall sub_8316D0(__m128i *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 i; // rax
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r8
  _DWORD *v8; // r14
  _DWORD *j; // rsi
  __int64 v10; // rax

  v3 = a1->m128i_i64[0];
  if ( !(unsigned int)sub_8D3A70(a1->m128i_i64[0]) )
  {
LABEL_18:
    if ( a1[1].m128i_i8[1] != 1 )
      return;
LABEL_21:
    sub_6ECC10((__int64)a1, a2);
    return;
  }
  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v5 = *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL);
  if ( v5 )
    v5 = sub_8D71D0(a2);
  v8 = (_DWORD *)sub_8D46C0(v5);
  for ( j = v8; *((_BYTE *)j + 140) == 12; j = (_DWORD *)*((_QWORD *)j + 20) )
  {
    j = (_DWORD *)*((_QWORD *)j + 20);
    if ( *((_BYTE *)j + 140) != 12 )
      break;
  }
  while ( *(_BYTE *)(v3 + 140) == 12 )
    v3 = *(_QWORD *)(v3 + 160);
  if ( (_DWORD *)v3 != j )
  {
    v6 = dword_4F07588;
    if ( !dword_4F07588 || (v10 = *(_QWORD *)(v3 + 32), *((_QWORD *)j + 4) != v10) || !v10 )
    {
      a2 = sub_8D5CE0(v3, j);
      if ( a2 )
      {
        sub_6F7270(a1, a2, v8, 0, 0, 1, 0, 1);
        goto LABEL_18;
      }
    }
  }
  a2 = (__int64)v8;
  sub_831640(a1, v8, 0, v6, v7);
  if ( a1[1].m128i_i8[1] == 1 )
    goto LABEL_21;
}
