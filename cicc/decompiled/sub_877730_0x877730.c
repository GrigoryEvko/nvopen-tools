// Function: sub_877730
// Address: 0x877730
//
__int64 __fastcall sub_877730(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  char v5; // al
  __int64 v6; // r12
  _QWORD *v7; // rbx
  _QWORD *v8; // rdi
  __int64 v9; // r15
  _QWORD *v11; // rax
  _QWORD *v12; // rdx

  v5 = *(_BYTE *)(a1 + 80);
  if ( v5 == 16 )
  {
    a1 = **(_QWORD **)(a1 + 88);
    v5 = *(_BYTE *)(a1 + 80);
  }
  if ( v5 == 24 )
    a1 = *(_QWORD *)(a1 + 88);
  v6 = 0;
  v7 = **(_QWORD ***)(*(_QWORD *)(a1 + 64) + 168LL);
  while ( v7 )
  {
    v8 = v7;
    v7 = (_QWORD *)*v7;
    if ( (v8[12] & 2) != 0 )
    {
      v9 = sub_8E5310(v8, a4, 0);
      if ( !v6 )
      {
        v11 = a3;
        do
        {
          v12 = v11;
          v11 = (_QWORD *)*v11;
        }
        while ( v11 );
        v6 = sub_8E5310(v12[2], a4, 0);
        if ( *(_BYTE *)(a2 + 80) == 16 )
          v6 = sub_8E5650(*(_QWORD *)(*(_QWORD *)(a2 + 88) + 8LL));
      }
      if ( v6 == v9 || (unsigned int)sub_8D5D50(v6, v9) )
        return 1;
    }
  }
  return 0;
}
