// Function: sub_688D10
// Address: 0x688d10
//
__int64 __fastcall sub_688D10(__int64 a1, int a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v11; // rax
  __int64 v12; // rax

  v8 = sub_726700(a1);
  v9 = sub_72BA30(unk_4F06A51);
  *(_BYTE *)(v8 + 56) = a2;
  *(_QWORD *)v8 = v9;
  if ( a2 )
  {
    *(_QWORD *)(v8 + 64) = a3;
  }
  else
  {
    if ( dword_4F04C44 != -1
      || (v11 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v11 + 6) & 6) != 0)
      || *(_BYTE *)(v11 + 4) == 12 )
    {
      if ( (unsigned int)sub_8DBE70(*a4) )
        sub_6F40C0(a4);
    }
    sub_6F6C80(a4);
    v12 = sub_6F6F40(a4, 0);
    *(_QWORD *)(v8 + 64) = v12;
    if ( (*(_BYTE *)(v12 + 25) & 1) != 0 && *(_BYTE *)(v12 + 24) == 3 )
      *(_BYTE *)(*(_QWORD *)(v12 + 56) + 88LL) |= 4u;
  }
  if ( a5 )
    sub_6E70E0(v8, a5);
  return v8;
}
