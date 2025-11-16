// Function: sub_18B19D0
// Address: 0x18b19d0
//
__int64 __fastcall sub_18B19D0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
    goto LABEL_16;
  while ( *(_UNKNOWN **)v3 != &unk_4F9D764 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_16;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9D764);
  v6 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 1432) = v5;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
    goto LABEL_16;
  while ( *(_UNKNOWN **)v7 != &unk_4F9D3C0 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_16;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F9D3C0);
  v10 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 1440) = v9;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F99CCD )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_16;
  }
  v13 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
                      *(_QWORD *)(v11 + 8),
                      &unk_4F99CCD)
                  + 160);
  if ( *(_BYTE *)(a1 + 1400) )
    return sub_18AE880(a1 + 160, a2, 0, v13);
  else
    return 0;
}
