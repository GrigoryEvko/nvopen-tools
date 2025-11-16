// Function: sub_1D03600
// Address: 0x1d03600
//
__int64 __fastcall sub_1D03600(_QWORD *a1)
{
  __int64 *v1; // r13
  __int64 *v2; // r12
  __int64 *v3; // rbx
  __int64 v5; // rdi
  __int64 *v6; // rsi
  unsigned __int8 v7; // dl
  unsigned __int8 v8; // al
  __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned int v11; // edx
  int v12; // ecx

  v1 = (__int64 *)a1[3];
  v2 = (__int64 *)a1[2];
  if ( v1 == v2 )
    return 0;
  v3 = v2 + 1;
  v5 = *v2;
  if ( v1 == v2 + 1 )
    goto LABEL_14;
  do
  {
    while ( 1 )
    {
      v6 = (__int64 *)*v3;
      v7 = (*(_BYTE *)(v5 + 229) & 0x10) != 0;
      v8 = (*(_BYTE *)(*v3 + 229) & 0x10) != 0;
      if ( v7 != v8 )
        break;
      v9 = *v6;
      if ( *(_QWORD *)v5 )
      {
        v10 = *(_DWORD *)(*(_QWORD *)v5 + 64LL);
        if ( !v9 )
        {
          v12 = *(_DWORD *)(*(_QWORD *)v5 + 64LL);
          v11 = 0;
          goto LABEL_7;
        }
LABEL_6:
        v11 = *(_DWORD *)(v9 + 64);
        v12 = v11 | v10;
LABEL_7:
        if ( v11 == v10 || !v12 )
          goto LABEL_21;
        if ( v10 && (v11 > v10 || !v11) )
        {
          v2 = v3++;
          v5 = (__int64)v6;
          if ( v1 == v3 )
            goto LABEL_13;
        }
        else if ( v1 == ++v3 )
        {
          goto LABEL_13;
        }
      }
      else
      {
        if ( v9 )
        {
          v10 = 0;
          goto LABEL_6;
        }
LABEL_21:
        if ( sub_1D03130(v5, (__int64)v6, a1[21]) )
        {
          v5 = *v3;
          v2 = v3++;
          if ( v1 == v3 )
            goto LABEL_13;
        }
        else
        {
          ++v3;
          v5 = *v2;
          if ( v1 == v3 )
            goto LABEL_13;
        }
      }
    }
    if ( v7 < v8 )
    {
      v5 = *v3;
      v2 = v3;
    }
    ++v3;
  }
  while ( v1 != v3 );
LABEL_13:
  v3 = (__int64 *)a1[3];
LABEL_14:
  if ( v2 != v3 - 1 )
  {
    *v2 = *(v3 - 1);
    *(v3 - 1) = v5;
    v2 = (__int64 *)(a1[3] - 8LL);
  }
  a1[3] = v2;
  *(_DWORD *)(v5 + 196) = 0;
  return v5;
}
