// Function: sub_8543B0
// Address: 0x8543b0
//
_DWORD *__fastcall sub_8543B0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _DWORD *result; // rax
  int v4; // r13d
  __int64 v5; // rax
  char v6; // si
  __int64 v7; // rax
  char v8[17]; // [rsp+Fh] [rbp-11h] BYREF

  result = &dword_4F04D80;
  v4 = dword_4F04D80;
  if ( !dword_4F04D80 )
  {
    if ( a2 )
    {
      v5 = sub_87D510(a2, v8);
      v6 = v8[0];
      a3 = v5;
    }
    else if ( a3 )
    {
      v8[0] = 21;
      v6 = 21;
    }
    else
    {
      v7 = a1[1];
      v8[0] = 0;
      v6 = 0;
      v4 = (*(_BYTE *)(v7 + 17) & 4) != 0;
    }
    return (_DWORD *)sub_8540F0(a1, v6, a3, v4);
  }
  return result;
}
