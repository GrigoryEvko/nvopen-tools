// Function: sub_22C0650
// Address: 0x22c0650
//
__int64 __fastcall sub_22C0650(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  int v3; // eax
  int v4; // eax

  result = *a2;
  *(_WORD *)a1 = (unsigned __int8)result;
  if ( (unsigned __int8)result > 3u )
  {
    result = (unsigned int)(result - 4);
    if ( (unsigned __int8)result <= 1u )
    {
      v3 = *((_DWORD *)a2 + 4);
      *((_DWORD *)a2 + 4) = 0;
      *(_DWORD *)(a1 + 16) = v3;
      *(_QWORD *)(a1 + 8) = *((_QWORD *)a2 + 1);
      v4 = *((_DWORD *)a2 + 8);
      *((_DWORD *)a2 + 8) = 0;
      *(_DWORD *)(a1 + 32) = v4;
      *(_QWORD *)(a1 + 24) = *((_QWORD *)a2 + 3);
      result = a2[1];
      *(_BYTE *)(a1 + 1) = result;
    }
    goto LABEL_4;
  }
  if ( (unsigned __int8)result <= 1u )
  {
LABEL_4:
    *a2 = 0;
    return result;
  }
  result = *((_QWORD *)a2 + 1);
  *a2 = 0;
  *(_QWORD *)(a1 + 8) = result;
  return result;
}
