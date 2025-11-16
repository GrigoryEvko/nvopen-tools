// Function: sub_1314420
// Address: 0x1314420
//
__int64 __fastcall sub_1314420(__int64 a1, _QWORD *a2, unsigned int a3)
{
  _QWORD *v4; // rdi
  __int64 v6; // rax
  __int64 v7; // rax

  v4 = (_QWORD *)a2[24];
  if ( v4 )
  {
    if ( ((*v4 >> 28) & 0x3FF) != 0 )
      return sub_13143B0(v4, (_QWORD *)&unk_5260DE0 + 5 * a3);
    if ( *(_DWORD *)(a1 + 78928) >= dword_5057900[0] )
    {
      v4[5] = v4;
      v4[6] = v4;
      v6 = a2[27];
      if ( v6 )
      {
        v4[5] = *(_QWORD *)(v6 + 48);
        *(_QWORD *)(a2[27] + 48LL) = v4;
        v4[6] = *(_QWORD *)(v4[6] + 40LL);
        *(_QWORD *)(*(_QWORD *)(a2[27] + 48LL) + 40LL) = a2[27];
        *(_QWORD *)(v4[6] + 40LL) = v4;
        v4 = (_QWORD *)v4[5];
      }
      a2[27] = v4;
    }
  }
  v7 = sub_133FAA0(a2 + 25);
  v4 = (_QWORD *)v7;
  if ( v7 )
  {
    ++a2[21];
    --a2[23];
    a2[24] = v7;
    return sub_13143B0(v4, (_QWORD *)&unk_5260DE0 + 5 * a3);
  }
  a2[24] = 0;
  return 0;
}
