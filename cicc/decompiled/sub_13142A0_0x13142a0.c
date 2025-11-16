// Function: sub_13142A0
// Address: 0x13142a0
//
__int64 __fastcall sub_13142A0(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 a4)
{
  _QWORD *v5; // rsi
  _QWORD *v6; // rbx
  int v7; // eax
  __int64 result; // rax

  v5 = (_QWORD *)a3[24];
  v6 = a3;
  if ( !v5 )
    goto LABEL_5;
  a4 = v5[1];
  a3 = *(_QWORD **)(a2 + 8);
  v7 = (v5[4] > *(_QWORD *)(a2 + 32)) - (v5[4] < *(_QWORD *)(a2 + 32));
  if ( v5[4] > *(_QWORD *)(a2 + 32) == v5[4] < *(_QWORD *)(a2 + 32) )
    v7 = (a4 > (unsigned __int64)a3) - (a4 < (unsigned __int64)a3);
  if ( v7 == 1 )
  {
    if ( ((*v5 >> 28) & 0x3FF) != 0 )
    {
      result = sub_133F890(v6 + 25, v5, a3, a4);
      ++v6[23];
    }
    else
    {
      result = dword_5057900[0];
      if ( *(_DWORD *)(a1 + 78928) >= dword_5057900[0] )
      {
        v5[5] = v5;
        v5[6] = v5;
        result = v6[27];
        if ( result )
        {
          v5[5] = *(_QWORD *)(result + 48);
          *(_QWORD *)(v6[27] + 48LL) = v5;
          v5[6] = *(_QWORD *)(v5[6] + 40LL);
          *(_QWORD *)(*(_QWORD *)(v6[27] + 48LL) + 40LL) = v6[27];
          result = v5[6];
          *(_QWORD *)(result + 40) = v5;
          v5 = (_QWORD *)v5[5];
        }
        v6[27] = v5;
      }
    }
    v6[24] = a2;
    ++v6[21];
  }
  else
  {
LABEL_5:
    result = sub_133F890(v6 + 25, a2, a3, a4);
    ++v6[23];
  }
  return result;
}
