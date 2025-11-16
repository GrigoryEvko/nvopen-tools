// Function: sub_73C8D0
// Address: 0x73c8d0
//
_QWORD *__fastcall sub_73C8D0(unsigned __int8 a1, unsigned __int64 a2)
{
  const __m128i *v2; // r14
  _QWORD *v3; // rax
  _QWORD *v4; // r12
  _QWORD *v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // rax

  if ( a2 > 0x50 || !dword_4F07588 || (v7 = qword_4F07BB0 + 648LL * a1, !*(_QWORD *)(v7 + 8 * a2)) )
  {
    v2 = (const __m128i *)sub_72C330(a1);
    if ( unk_4D047E0 )
      v2 = sub_73C570(v2, 1);
    v3 = sub_7259C0(8);
    v3[20] = v2;
    v4 = v3;
    v3[22] = a2;
    if ( a2 )
    {
      sub_8D6090(v3);
      if ( a2 > 0x50 )
      {
        if ( *(v4 - 2) )
          return v4;
        goto LABEL_12;
      }
    }
    else
    {
      *((_BYTE *)v3 + 169) |= 0x20u;
      sub_8D6090(v3);
    }
    if ( !dword_4F07588 )
    {
      if ( *(v4 - 2) )
        return v4;
      goto LABEL_17;
    }
    *(_QWORD *)(qword_4F07BB0 + 648LL * a1 + 8 * a2) = v4;
    if ( *(v4 - 2) )
      return v4;
LABEL_12:
    if ( dword_4F07588 )
    {
      v6 = *(_QWORD **)(unk_4D03FF0 + 376LL);
      goto LABEL_18;
    }
LABEL_17:
    v6 = &unk_4F06D00;
LABEL_18:
    v8 = (_QWORD *)v6[13];
    if ( v4 != v8 )
    {
      if ( v8 )
        *(v8 - 2) = v4;
      else
        v6[12] = v4;
      v6[13] = v4;
    }
    return v4;
  }
  return *(_QWORD **)(v7 + 8 * a2);
}
