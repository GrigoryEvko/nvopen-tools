// Function: sub_24C5360
// Address: 0x24c5360
//
__int64 __fastcall sub_24C5360(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v3; // eax
  __int64 v5; // rbx
  int v6; // eax

  v3 = *(_DWORD *)(a2 + 604);
  if ( v3 == 1 )
  {
    v5 = a1 + 16;
    if ( sub_2241AC0((__int64)a3, "sancov_cntrs") )
    {
      if ( sub_2241AC0((__int64)a3, "sancov_bools") )
      {
        v6 = sub_2241AC0((__int64)a3, "sancov_pcs");
        *(_QWORD *)a1 = v5;
        if ( v6 )
          strcpy((char *)(a1 + 16), ".SCOV$GM");
        else
          strcpy((char *)(a1 + 16), ".SCOVP$M");
        *(_QWORD *)(a1 + 8) = 8;
      }
      else
      {
        *(_QWORD *)a1 = v5;
        strcpy((char *)(a1 + 16), ".SCOV$BM");
        *(_QWORD *)(a1 + 8) = 8;
      }
      return a1;
    }
    *(_QWORD *)a1 = v5;
    strcpy((char *)(a1 + 16), ".SCOV$CM");
    *(_QWORD *)(a1 + 8) = 8;
    return a1;
  }
  else
  {
    if ( v3 != 5 )
    {
      sub_8FD6D0(a1, "__", a3);
      return a1;
    }
    sub_8FD6D0(a1, "__DATA,__", a3);
    return a1;
  }
}
