// Function: sub_335E330
// Address: 0x335e330
//
__int64 __fastcall sub_335E330(__int64 a1)
{
  __int64 result; // rax
  int v2; // edx
  __int64 v3; // rcx
  unsigned int v4; // edx

  result = *(_QWORD *)(a1 + 8);
  if ( !result )
    return result;
  v2 = *(_DWORD *)(result + 24);
  if ( v2 >= 0 )
  {
    if ( v2 == 50 )
    {
      *(_DWORD *)(a1 + 20) = 1;
      return result;
    }
LABEL_10:
    *(_DWORD *)(a1 + 20) = 0;
    return result;
  }
  v3 = (unsigned int)~v2;
  if ( v2 == -11 || (_DWORD)v3 == 28 && **(_WORD **)(result + 48) == 1 )
    goto LABEL_10;
  result = *(unsigned int *)(result + 68);
  v4 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) + 8LL) - 40 * v3 + 4);
  *(_DWORD *)(a1 + 16) = 0;
  if ( v4 > (unsigned int)result )
    v4 = result;
  *(_DWORD *)(a1 + 20) = v4;
  return result;
}
