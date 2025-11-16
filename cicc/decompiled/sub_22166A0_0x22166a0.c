// Function: sub_22166A0
// Address: 0x22166a0
//
__int64 __fastcall sub_22166A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  size_t v4; // rdx
  __int64 v5; // rbp
  __int64 v6; // r12

  v3 = sub_2216040(a3 + *(_QWORD *)a1, *(_QWORD *)(a1 + 8));
  v4 = *(_QWORD *)a1;
  v5 = v3;
  v6 = v3 + 24;
  if ( *(_QWORD *)a1 )
  {
    if ( v4 == 1 )
    {
      *(_DWORD *)(v3 + 24) = *(_DWORD *)(a1 + 24);
      if ( (_UNKNOWN *)v3 == &unk_4FD67E0 )
        return v6;
      goto LABEL_7;
    }
    wmemcpy((wchar_t *)(v3 + 24), (const wchar_t *)(a1 + 24), v4);
    v4 = *(_QWORD *)a1;
  }
  if ( (_UNKNOWN *)v5 == &unk_4FD67E0 )
    return v6;
LABEL_7:
  *(_DWORD *)(v5 + 16) = 0;
  *(_QWORD *)v5 = v4;
  *(_DWORD *)(v5 + 4 * v4 + 24) = 0;
  return v6;
}
