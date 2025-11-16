// Function: sub_264E780
// Address: 0x264e780
//
__int64 __fastcall sub_264E780(__int64 a1, __int64 *a2, char a3)
{
  __int64 v5; // rdi
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 result; // rax

  v5 = a1 + 24;
  v6 = *(_QWORD *)(v5 - 24);
  v7 = *(_QWORD *)(v5 - 16);
  sub_264E600(v5);
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = 0;
  if ( a2 )
  {
    if ( a3 )
    {
      sub_2647750(v6, a1);
      result = sub_26476C0(v7 + 48, *a2);
    }
    else
    {
      sub_2647840(v7, a1);
      result = sub_26476C0(v6 + 72, *a2);
    }
    *a2 = result;
  }
  else
  {
    sub_2647750(v6, a1);
    return sub_2647840(v7, a1);
  }
  return result;
}
