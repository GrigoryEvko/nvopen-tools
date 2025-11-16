// Function: sub_80A3E0
// Address: 0x80a3e0
//
void __fastcall sub_80A3E0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rcx
  _QWORD *v6; // rax
  __int64 v7; // rcx

  if ( (*(_BYTE *)(a1 + 201) & 0x10) == 0 )
  {
    v3 = (_QWORD *)qword_4F18B98;
    if ( qword_4F18B98 )
    {
      v4 = *(_QWORD *)qword_4F18B98;
      *(_QWORD *)qword_4F18B98 = 0;
      qword_4F18B98 = v4;
      v5 = *(_QWORD *)(a3 + 8);
      v3[1] = a1;
      *v3 = v5;
      *(_QWORD *)(a3 + 8) = v3;
    }
    else
    {
      v6 = sub_725220();
      v7 = *(_QWORD *)(a3 + 8);
      v6[1] = a1;
      *v6 = v7;
      *(_QWORD *)(a3 + 8) = v6;
    }
  }
}
