// Function: sub_1322200
// Address: 0x1322200
//
__int64 __fastcall sub_1322200(_BYTE *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned int v5; // r12d
  __int64 v7; // rdx

  v2 = *(_QWORD *)(qword_4F96BA0 + 16);
  if ( !v2 || (v3 = *(_QWORD *)(v2 + 16)) == 0 )
  {
    v5 = *(_DWORD *)(qword_4F96BA0 + 8);
    goto LABEL_7;
  }
  if ( v2 == v3 )
  {
    v7 = *(_QWORD *)(v3 + 8);
    if ( v7 == v3 )
    {
      *(_QWORD *)(qword_4F96BA0 + 16) = 0;
      goto LABEL_5;
    }
    *(_QWORD *)(qword_4F96BA0 + 16) = v7;
  }
  *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL) = *(_QWORD *)(*(_QWORD *)(v3 + 8) + 16LL);
  v4 = *(_QWORD *)(v3 + 16);
  *(_QWORD *)(*(_QWORD *)(v3 + 8) + 16LL) = v4;
  *(_QWORD *)(v3 + 16) = *(_QWORD *)(v4 + 8);
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 8) + 16LL) + 8LL) = *(_QWORD *)(v3 + 8);
  *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL) = v3;
LABEL_5:
  v5 = *(_DWORD *)v3;
LABEL_7:
  if ( !sub_1322110(a1, v5, 0, 1) || !sub_1300B80((__int64)a1, v5, a2) )
    return 0xFFFFFFFFLL;
  if ( *(_DWORD *)(qword_4F96BA0 + 8) == v5 )
    *(_DWORD *)(qword_4F96BA0 + 8) = v5 + 1;
  return v5;
}
