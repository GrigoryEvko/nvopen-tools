// Function: sub_9914A0
// Address: 0x9914a0
//
unsigned __int64 __fastcall sub_9914A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // r12
  __int64 v6; // rax
  char v7; // bl

  v3 = *(_QWORD *)(a1 + 8);
  v4 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v3 )
    goto LABEL_6;
  v5 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = (v3 >> 1) & 3;
  if ( v6 != 2 )
  {
    if ( v6 == 1 && v4 )
    {
      v5 = *(_QWORD *)(v4 + 24);
      return (unsigned __int64)(sub_9208B0(a2, v5) + 7) >> 3;
    }
LABEL_6:
    v5 = sub_BCBAE0(v4, **(_QWORD **)a1);
    if ( ((*(__int64 *)(a1 + 8) >> 1) & 3) != 1 )
      goto LABEL_4;
    return (unsigned __int64)(sub_9208B0(a2, v5) + 7) >> 3;
  }
  if ( !v4 )
    goto LABEL_6;
LABEL_4:
  v7 = sub_AE5020(a2, v5);
  return ((1LL << v7) + ((unsigned __int64)(sub_9208B0(a2, v5) + 7) >> 3) - 1) >> v7 << v7;
}
