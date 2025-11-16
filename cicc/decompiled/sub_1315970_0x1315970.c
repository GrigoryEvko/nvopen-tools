// Function: sub_1315970
// Address: 0x1315970
//
void __fastcall sub_1315970(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v5; // rax
  _QWORD *v6; // rax

  if ( a3 == (_QWORD *)a4[24] )
  {
    a4[24] = 0;
    --a4[22];
  }
  else
  {
    if ( *((_DWORD *)&unk_5260DE0 + 10 * (unsigned __int8)(*a3 >> 20) + 4) == 1 )
    {
      if ( *(_DWORD *)(a2 + 78928) >= dword_5057900[0] )
      {
        if ( a3 != (_QWORD *)a4[27] )
        {
LABEL_5:
          *(_QWORD *)(a3[6] + 40LL) = *(_QWORD *)(a3[5] + 48LL);
          v5 = a3[6];
          *(_QWORD *)(a3[5] + 48LL) = v5;
          a3[6] = *(_QWORD *)(v5 + 40);
          *(_QWORD *)(*(_QWORD *)(a3[5] + 48LL) + 40LL) = a3[5];
          *(_QWORD *)(a3[6] + 40LL) = a3;
          --a4[22];
          return;
        }
        v6 = (_QWORD *)a3[5];
        if ( a3 != v6 )
        {
          a4[27] = v6;
          goto LABEL_5;
        }
        a4[27] = 0;
      }
    }
    else
    {
      sub_1340070(a4 + 25, a3);
      --a4[23];
    }
    --a4[22];
  }
}
