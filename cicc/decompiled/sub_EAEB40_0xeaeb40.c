// Function: sub_EAEB40
// Address: 0xeaeb40
//
__int64 __fastcall sub_EAEB40(__int64 *a1)
{
  __int64 v2; // rdi
  bool v3; // zf
  unsigned int v4; // r12d
  _QWORD *v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+8h] [rbp-38h]
  _QWORD v8[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = *a1;
  v6 = v8;
  v7 = 0;
  v3 = *(_BYTE *)(v2 + 869) == 0;
  LOBYTE(v8[0]) = 0;
  if ( v3 )
  {
    if ( !(unsigned __int8)sub_EA2540(v2) )
    {
      v2 = *a1;
      goto LABEL_8;
    }
  }
  else
  {
LABEL_8:
    while ( !(unsigned __int8)sub_EAE3B0((_QWORD *)v2, &v6) )
    {
      (*(void (__fastcall **)(_QWORD, _QWORD *, __int64))(**(_QWORD **)(*a1 + 232) + 512LL))(
        *(_QWORD *)(*a1 + 232),
        v6,
        v7);
      if ( *(_BYTE *)a1[1] )
        goto LABEL_14;
      if ( *(_DWORD *)sub_ECD7B0(*a1) != 3 )
      {
        v4 = *(unsigned __int8 *)a1[1];
        if ( !(_BYTE)v4 )
          goto LABEL_10;
LABEL_14:
        v4 = 0;
        (*(void (__fastcall **)(_QWORD, void *, __int64))(**(_QWORD **)(*a1 + 232) + 512LL))(
          *(_QWORD *)(*a1 + 232),
          &unk_3F83918,
          1);
        goto LABEL_10;
      }
      v2 = *a1;
    }
  }
  v4 = 1;
LABEL_10:
  if ( v6 != v8 )
    j_j___libc_free_0(v6, v8[0] + 1LL);
  return v4;
}
