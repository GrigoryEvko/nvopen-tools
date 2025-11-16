// Function: sub_13E09D0
// Address: 0x13e09d0
//
__int64 __fastcall sub_13E09D0(_QWORD *a1, __int64 a2, __int64 *a3, int a4)
{
  int v6; // eax
  int v9; // eax
  __int64 **v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdi

  v6 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v6 > 0x17u )
  {
    v9 = v6 - 24;
LABEL_6:
    if ( v9 == 38 )
    {
      v10 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
          ? *(__int64 ***)(a2 - 8)
          : (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v11 = *v10;
      if ( v11 )
      {
        v12 = *v11;
        if ( *(_BYTE *)(*v11 + 8) == 16 )
          v12 = **(_QWORD **)(v12 + 16);
        if ( (unsigned __int8)sub_1642F90(v12, 1) )
          return sub_15A06D0(*a1);
      }
    }
    goto LABEL_3;
  }
  if ( (_BYTE)v6 == 5 )
  {
    v9 = *(unsigned __int16 *)(a2 + 18);
    goto LABEL_6;
  }
LABEL_3:
  if ( !(unsigned __int8)sub_14B0710(a1, a2, 0) )
    return sub_13E0700(21, a1, (unsigned __int8 *)a2, a3, a4);
  return sub_15A06D0(*a1);
}
