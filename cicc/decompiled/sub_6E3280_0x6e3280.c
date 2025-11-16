// Function: sub_6E3280
// Address: 0x6e3280
//
__int64 __fastcall sub_6E3280(__int64 a1, _DWORD *a2)
{
  _QWORD *v2; // r12
  __int64 result; // rax

  v2 = a2;
  result = sub_6DEE30(a1);
  if ( result )
  {
    if ( a2 )
    {
      if ( !*a2 )
      {
LABEL_4:
        *(_QWORD *)(result + 28) = *(_QWORD *)(a1 + 68);
LABEL_5:
        *(_QWORD *)(result + 36) = *(_QWORD *)(a1 + 68);
        *(_QWORD *)(result + 44) = *(_QWORD *)(a1 + 76);
        return result;
      }
    }
    else
    {
      if ( !*(_DWORD *)(result + 28) )
        goto LABEL_4;
      v2 = (_QWORD *)(result + 28);
    }
    *(_QWORD *)(result + 28) = *v2;
    goto LABEL_5;
  }
  return result;
}
