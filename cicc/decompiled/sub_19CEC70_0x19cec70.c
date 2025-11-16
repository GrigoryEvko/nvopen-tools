// Function: sub_19CEC70
// Address: 0x19cec70
//
__int64 __fastcall sub_19CEC70(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx

  v1 = *(__int64 **)(*(_QWORD *)a1 + 8LL);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9E06C )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_6;
  }
  return (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9E06C)
       + 160;
}
