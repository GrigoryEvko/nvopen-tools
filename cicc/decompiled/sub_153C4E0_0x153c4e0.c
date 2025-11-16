// Function: sub_153C4E0
// Address: 0x153c4e0
//
__int64 __fastcall sub_153C4E0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rcx
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx

  if ( *(_BYTE *)(a1 + 169) )
  {
    v4 = *(__int64 **)(a1 + 8);
    v5 = *v4;
    v6 = v4[1];
    if ( v5 == v6 )
LABEL_9:
      BUG();
    while ( *(_UNKNOWN **)v5 != &unk_4F9994C )
    {
      v5 += 16;
      if ( v6 == v5 )
        goto LABEL_9;
    }
    v2 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F9994C)
       + 160;
  }
  else
  {
    v2 = 0;
  }
  sub_153BF40(a2, *(_QWORD *)(a1 + 160), *(_BYTE *)(a1 + 168), v2, *(_BYTE *)(a1 + 170), 0);
  return 0;
}
