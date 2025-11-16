// Function: sub_31DB840
// Address: 0x31db840
//
char __fastcall sub_31DB840(__int64 a1, _QWORD *a2)
{
  int v2; // r13d
  unsigned __int64 v3; // rax
  __int64 v4; // r9
  __int64 v5; // rax
  __int16 v6; // cx
  unsigned __int16 v7; // cx

  v2 = *(_DWORD *)(*(_QWORD *)(a1 + 208) + 336LL);
  LOBYTE(v3) = sub_31DB810(a1);
  if ( (_BYTE)v3 || (v2 & 0xFFFFFFFD) == 1 )
  {
    LODWORD(v3) = sub_31DB780(a1, *(__int64 **)(a1 + 232));
    if ( (_DWORD)v3 )
    {
      v4 = a2[3];
      v5 = a2[1];
      if ( v4 + 48 == v5 )
      {
LABEL_13:
        v3 = *(_QWORD *)(*(_QWORD *)(v4 + 32) + 320LL) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v4 == v3 )
          return v3;
      }
      else
      {
        while ( 1 )
        {
          v6 = *(_WORD *)(v5 + 68);
          if ( v6 )
          {
            v7 = v6 - 9;
            if ( (v7 > 0x3Bu || ((1LL << v7) & 0x800000000000C09LL) == 0)
              && (*(_BYTE *)(*(_QWORD *)(v5 + 16) + 24LL) & 0x10) == 0 )
            {
              break;
            }
          }
          v5 = *(_QWORD *)(v5 + 8);
          if ( v4 + 48 == v5 )
            goto LABEL_13;
        }
      }
      LOBYTE(v3) = sub_31F1030(a1, *(_QWORD *)(*(_QWORD *)(a1 + 232) + 360LL) + 104LL * *(unsigned int *)(a2[4] + 24LL));
    }
  }
  return v3;
}
