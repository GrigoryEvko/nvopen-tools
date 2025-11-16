// Function: sub_111EDC0
// Address: 0x111edc0
//
__int64 __fastcall sub_111EDC0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned int v7; // r12d
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 64);
  if ( !v2 )
  {
    v3 = *(_QWORD *)(a2 - 32);
    if ( !v3 )
      return 0;
    goto LABEL_4;
  }
  *(_QWORD *)a1[1] = v2;
  v3 = *(_QWORD *)(a2 - 32);
  v4 = *(_QWORD *)(v3 + 16);
  if ( !v4
    || *(_QWORD *)(v4 + 8)
    || *(_BYTE *)v3 != 85
    || (v9 = *(_QWORD *)(v3 - 32)) == 0
    || *(_BYTE *)v9
    || *(_QWORD *)(v9 + 24) != *(_QWORD *)(v3 + 80)
    || *(_DWORD *)(v9 + 36) != *((_DWORD *)a1 + 4)
    || (v10 = *(_DWORD *)(v3 + 4) & 0x7FFFFFF,
        *(_QWORD *)(v3 + 32 * (*((unsigned int *)a1 + 6) - v10)) != *(_QWORD *)a1[4])
    || *(_QWORD *)(v3 + 32 * (*((unsigned int *)a1 + 10) - v10)) != *(_QWORD *)a1[6]
    || (v11 = *(_QWORD *)(v3 + 32 * (*((unsigned int *)a1 + 14) - v10))) == 0 )
  {
LABEL_4:
    *(_QWORD *)a1[1] = v3;
    v5 = *(_QWORD *)(a2 - 64);
    v6 = *(_QWORD *)(v5 + 16);
    if ( v6 )
    {
      if ( !*(_QWORD *)(v6 + 8) && *(_BYTE *)v5 == 85 )
      {
        v14 = *(_QWORD *)(v5 - 32);
        if ( v14 )
        {
          if ( !*(_BYTE *)v14
            && *(_QWORD *)(v14 + 24) == *(_QWORD *)(v5 + 80)
            && *(_DWORD *)(v14 + 36) == *((_DWORD *)a1 + 4) )
          {
            v15 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
            if ( *(_QWORD *)a1[4] == *(_QWORD *)(v5 + 32 * (*((unsigned int *)a1 + 6) - v15))
              && *(_QWORD *)a1[6] == *(_QWORD *)(v5 + 32 * (*((unsigned int *)a1 + 10) - v15)) )
            {
              v16 = *(_QWORD *)(v5 + 32 * (*((unsigned int *)a1 + 14) - v15));
              if ( v16 )
              {
                v7 = 1;
                *(_QWORD *)a1[8] = v16;
                if ( *a1 )
                {
                  v17 = sub_B53960(a2);
                  v18 = *a1;
                  *(_DWORD *)v18 = v17;
                  *(_BYTE *)(v18 + 4) = BYTE4(v17);
                }
                return v7;
              }
            }
          }
        }
      }
    }
    return 0;
  }
  *(_QWORD *)a1[8] = v11;
  if ( *a1 )
  {
    v12 = sub_B53900(a2);
    v13 = *a1;
    *(_DWORD *)v13 = v12;
    *(_BYTE *)(v13 + 4) = BYTE4(v12);
  }
  return 1;
}
