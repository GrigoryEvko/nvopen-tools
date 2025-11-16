// Function: sub_87ADD0
// Address: 0x87add0
//
_BOOL8 __fastcall sub_87ADD0(__int64 a1, _DWORD *a2, _DWORD *a3, _DWORD *a4, __int64 a5)
{
  __int64 i; // rax
  __int64 v7; // rax
  _QWORD *v8; // r15
  _QWORD *v9; // r15
  __int64 j; // rbx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  _DWORD *v18; // [rsp+8h] [rbp-38h]

  *a2 = 0;
  *a3 = 0;
  *a4 = 0;
  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = *(_QWORD *)(i + 168);
  if ( (*(_BYTE *)(v7 + 16) & 1) != 0 )
    return 0;
  v8 = *(_QWORD **)v7;
  if ( unk_4D04814 )
  {
    if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    {
      v14 = v8[1];
      v18 = a4;
      v15 = sub_72D2E0(*(_QWORD **)(*(_QWORD *)(a1 + 40) + 32LL));
      if ( (unsigned int)sub_8D97D0(v15, v14, 0, v16, v17) )
      {
        if ( (unsigned int)sub_8D3D00(*(_QWORD *)(*v8 + 8LL)) )
        {
          a4 = v18;
          *v18 = 1;
          v8 = (_QWORD *)*v8;
        }
      }
    }
  }
  v9 = (_QWORD *)*v8;
  if ( v9 )
  {
    for ( j = v9[1]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    if ( (*(_BYTE *)(a1 + 89) & 4) == 0
      && (!unk_4D04478 || (v11 = *(_QWORD *)(a1 + 40)) != 0 && *(_BYTE *)(v11 + 28) == 3)
      || !(unsigned int)sub_8D2780(j)
      || *(_BYTE *)(j + 160) != byte_4F06A51[0]
      || (*a2 = 1, (v9 = (_QWORD *)*v9) != 0) )
    {
      if ( dword_4D04818 )
      {
        v12 = v9[1];
        if ( v12 == unk_4F06C60 || (unsigned int)sub_8D97D0(v12, unk_4F06C60, 0, a4, a5) )
        {
          *a3 = 1;
          return *v9 == 0;
        }
      }
      return 0;
    }
  }
  return 1;
}
