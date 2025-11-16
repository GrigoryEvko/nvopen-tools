// Function: sub_7ABAA0
// Address: 0x7abaa0
//
__int64 __fastcall sub_7ABAA0(int a1, __int64 a2)
{
  __int64 result; // rax

  result = (__int64)qword_4F06448;
  if ( qword_4F06448 )
    qword_4F06448 = (_QWORD *)*qword_4F06448;
  else
    result = sub_823970(32);
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = a2;
  *(_DWORD *)(result + 16) = a1;
  *(_BYTE *)(result + 20) = 0;
  if ( a1 != 3 )
  {
    if ( !a1 )
    {
      *(_BYTE *)(result + 24) = 32;
      if ( unk_4F06458 )
        goto LABEL_5;
LABEL_9:
      unk_4F06458 = result;
      goto LABEL_6;
    }
    *(_DWORD *)(result + 24) = 0;
  }
  if ( !unk_4F06458 )
    goto LABEL_9;
LABEL_5:
  *qword_4F06450 = result;
LABEL_6:
  qword_4F06450 = (_QWORD *)result;
  dword_4F17FA0 = 0;
  return result;
}
