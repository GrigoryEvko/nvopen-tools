// Function: sub_C1D080
// Address: 0xc1d080
//
__int64 __fastcall sub_C1D080(__int64 a1, unsigned int *a2)
{
  __int64 v3; // rbx
  unsigned int v4; // ecx
  __int64 v5; // rax
  char v6; // si
  unsigned int v7; // edx
  bool v8; // zf
  __int64 v9; // r8
  __int64 v11; // rax
  unsigned int v12; // edx

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
  {
    v3 = a1 + 8;
    goto LABEL_17;
  }
  v4 = *a2;
  while ( 1 )
  {
    v7 = *(_DWORD *)(v3 + 32);
    if ( v7 > v4 )
    {
      v5 = *(_QWORD *)(v3 + 16);
      v6 = 1;
      goto LABEL_8;
    }
    if ( v7 == v4 && *(_DWORD *)(v3 + 36) > a2[1] )
      break;
    v5 = *(_QWORD *)(v3 + 24);
    v6 = 0;
    if ( !v5 )
      goto LABEL_9;
LABEL_5:
    v3 = v5;
  }
  v5 = *(_QWORD *)(v3 + 16);
  v6 = 1;
LABEL_8:
  if ( v5 )
    goto LABEL_5;
LABEL_9:
  if ( !v6 )
  {
    v8 = v7 == v4;
    if ( v7 >= v4 )
    {
LABEL_11:
      if ( !v8 || *(_DWORD *)(v3 + 36) >= a2[1] )
        return v3;
    }
    return 0;
  }
LABEL_17:
  v9 = 0;
  if ( v3 != *(_QWORD *)(a1 + 24) )
  {
    v11 = sub_220EF80(v3);
    v12 = *(_DWORD *)(v11 + 32);
    v3 = v11;
    v8 = v12 == *a2;
    if ( v12 >= *a2 )
      goto LABEL_11;
    return 0;
  }
  return v9;
}
