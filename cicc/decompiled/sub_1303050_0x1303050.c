// Function: sub_1303050
// Address: 0x1303050
//
__int64 __fastcall sub_1303050(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v4; // edx
  unsigned int v5; // eax
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdx

  v2 = a2;
  if ( a2 )
    return v2;
  if ( *(char *)(a1 + 1) <= 0 )
  {
    v2 = *(_QWORD *)(a1 + 144);
    if ( !v2 )
    {
      v2 = sub_1302AE0(a1, 0);
      if ( *(_BYTE *)a1 )
      {
        v7 = *(_QWORD *)(a1 + 296);
        v8 = a1 + 256;
        v9 = a1 + 856;
        if ( v7 )
        {
          if ( v2 != v7 )
            sub_1311F50(a1, v8, v9, v2);
        }
        else
        {
          sub_13114E0(a1, v8, v9, v2);
        }
      }
    }
    if ( unk_4C6F238 > 2u )
    {
      v4 = dword_505F9BC;
      if ( unk_4C6F238 == 4 && dword_505F9BC > 1u )
        v4 = (dword_505F9BC >> 1) - (((dword_505F9BC & 1) == 0) - 1);
      if ( *(_DWORD *)(v2 + 78928) < v4 && a1 != *(_QWORD *)(v2 + 16) )
      {
        v5 = sched_getcpu();
        if ( unk_4C6F238 != 3 && dword_505F9BC >> 1 <= v5 )
          v5 -= dword_505F9BC >> 1;
        if ( *(_DWORD *)(v2 + 78928) != v5 )
        {
          v2 = *(_QWORD *)(a1 + 144);
          if ( v5 != *(_DWORD *)(v2 + 78928) )
          {
            v6 = qword_50579C0[v5];
            if ( !v6 )
              v6 = sub_1300B80(a1, v5, (__int64)&off_49E8000);
            sub_1302A70(a1, v2, v6);
            if ( *(_BYTE *)a1 )
              sub_1311F50(a1, a1 + 256, a1 + 856, v6);
            v2 = *(_QWORD *)(a1 + 144);
          }
        }
        *(_QWORD *)(v2 + 16) = a1;
      }
    }
    return v2;
  }
  v2 = qword_50579C0[0];
  if ( qword_50579C0[0] )
    return v2;
  return sub_1300B80(a1, 0, (__int64)&off_49E8000);
}
