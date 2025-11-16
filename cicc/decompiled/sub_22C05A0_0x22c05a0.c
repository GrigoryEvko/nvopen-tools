// Function: sub_22C05A0
// Address: 0x22c05a0
//
__int64 __fastcall sub_22C05A0(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  unsigned int v3; // eax
  unsigned int v4; // eax
  unsigned int v5; // eax

  result = *a2;
  *(_WORD *)a1 = (unsigned __int8)result;
  if ( (unsigned __int8)result > 3u )
  {
    result = (unsigned int)(result - 4);
    if ( (unsigned __int8)result > 1u )
      return result;
    v3 = *((_DWORD *)a2 + 4);
    *(_DWORD *)(a1 + 16) = v3;
    if ( v3 > 0x40 )
    {
      sub_C43780(a1 + 8, (const void **)a2 + 1);
      v5 = *((_DWORD *)a2 + 8);
      *(_DWORD *)(a1 + 32) = v5;
      if ( v5 <= 0x40 )
        goto LABEL_5;
    }
    else
    {
      *(_QWORD *)(a1 + 8) = *((_QWORD *)a2 + 1);
      v4 = *((_DWORD *)a2 + 8);
      *(_DWORD *)(a1 + 32) = v4;
      if ( v4 <= 0x40 )
      {
LABEL_5:
        *(_QWORD *)(a1 + 24) = *((_QWORD *)a2 + 3);
LABEL_6:
        result = a2[1];
        *(_BYTE *)(a1 + 1) = result;
        return result;
      }
    }
    sub_C43780(a1 + 24, (const void **)a2 + 3);
    goto LABEL_6;
  }
  if ( (unsigned __int8)result > 1u )
  {
    result = *((_QWORD *)a2 + 1);
    *(_QWORD *)(a1 + 8) = result;
  }
  return result;
}
