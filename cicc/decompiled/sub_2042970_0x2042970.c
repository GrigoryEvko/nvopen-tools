// Function: sub_2042970
// Address: 0x2042970
//
__int64 __fastcall sub_2042970(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  unsigned int v3; // edx
  __int16 v4; // cx
  int v5; // ecx
  unsigned int *v6; // rcx

  result = *a2;
  v3 = 0;
  while ( 1 )
  {
    if ( !result )
    {
LABEL_8:
      *((_WORD *)a2 + 112) = v3;
      return result;
    }
    v4 = *(_WORD *)(result + 24);
    if ( v4 < 0 )
      break;
    if ( v4 == 47 )
      ++v3;
    else
      v3 += v4 == 193;
LABEL_6:
    v5 = *(_DWORD *)(result + 56);
    if ( v5 )
    {
      v6 = (unsigned int *)(*(_QWORD *)(result + 32) + 40LL * (unsigned int)(v5 - 1));
      result = *(_QWORD *)v6;
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * v6[2]) == 111 )
        continue;
    }
    goto LABEL_8;
  }
  if ( v4 != -10 )
  {
    v3 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(a1 + 144) + 8LL) + ((__int64)~v4 << 6) + 4);
    if ( *(_DWORD *)(result + 60) <= v3 )
      v3 = *(_DWORD *)(result + 60);
    goto LABEL_6;
  }
  *((_WORD *)a2 + 112) = 0;
  return result;
}
