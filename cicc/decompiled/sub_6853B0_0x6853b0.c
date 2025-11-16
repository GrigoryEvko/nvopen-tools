// Function: sub_6853B0
// Address: 0x6853b0
//
__int64 __fastcall sub_6853B0(unsigned __int8 a1, unsigned int a2, FILE *a3, __int64 a4)
{
  _DWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx

  v8 = sub_67D610(a2, a3, a1);
  if ( a4 )
  {
    v9 = sub_67BB20(4);
    *(_QWORD *)(v9 + 16) = a4;
    *(_DWORD *)(v9 + 24) = -1;
    if ( *((_QWORD *)v8 + 23) )
    {
      v10 = *((_QWORD *)v8 + 24);
      if ( !v10 )
      {
LABEL_5:
        *((_QWORD *)v8 + 24) = v9;
        return sub_6837D0((__int64)v8, a3);
      }
    }
    else
    {
      v10 = *((_QWORD *)v8 + 24);
      *((_QWORD *)v8 + 23) = v9;
      if ( !v10 )
        goto LABEL_5;
    }
    *(_QWORD *)(v10 + 8) = v9;
    goto LABEL_5;
  }
  return sub_6837D0((__int64)v8, a3);
}
