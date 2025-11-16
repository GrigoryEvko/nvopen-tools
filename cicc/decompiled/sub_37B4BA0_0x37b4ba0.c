// Function: sub_37B4BA0
// Address: 0x37b4ba0
//
__int64 __fastcall sub_37B4BA0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  unsigned int v3; // ecx
  int v4; // edx
  int v5; // edx
  unsigned int *v6; // rdx

  result = *a2;
  v3 = 0;
  while ( 1 )
  {
    if ( !result )
    {
LABEL_8:
      *((_WORD *)a2 + 125) = v3;
      return result;
    }
    v4 = *(_DWORD *)(result + 24);
    if ( v4 < 0 )
      break;
    if ( v4 == 50 )
      ++v3;
    else
      v3 += (unsigned int)(v4 - 307) < 2;
LABEL_6:
    v5 = *(_DWORD *)(result + 64);
    if ( v5 )
    {
      v6 = (unsigned int *)(*(_QWORD *)(result + 40) + 40LL * (unsigned int)(v5 - 1));
      result = *(_QWORD *)v6;
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v6 + 48LL) + 16LL * v6[2]) == 262 )
        continue;
    }
    goto LABEL_8;
  }
  if ( v4 != -11 )
  {
    v3 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(a1 + 144) + 8LL) - 40LL * (unsigned int)~v4 + 4);
    if ( *(_DWORD *)(result + 68) <= v3 )
      v3 = *(_DWORD *)(result + 68);
    goto LABEL_6;
  }
  *((_WORD *)a2 + 125) = 0;
  return result;
}
