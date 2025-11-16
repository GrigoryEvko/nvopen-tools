// Function: sub_650B30
// Address: 0x650b30
//
__int64 __fastcall sub_650B30(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r15
  __int64 i; // rax
  __int64 v13; // rdi
  __int64 result; // rax
  __int64 *v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rdi
  _QWORD *v18; // rbx
  __int64 v19; // r13
  __int64 v20; // rdi
  int v21; // eax
  bool v22; // zf
  __int64 v23; // r14
  __int64 v24; // rcx
  __int64 v25; // r8

  v5 = a2;
  v11 = sub_72BA30(5);
  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v13 = *(_QWORD *)(i + 160);
  if ( v11 != v13 && !(unsigned int)sub_8D97D0(v13, v11, 0, v9, v10) )
  {
    v20 = 5;
    if ( dword_4D04964 )
      v20 = unk_4F07471;
    sub_684AA0(v20, 951, a5);
  }
  if ( dword_4F077C4 == 2 || unk_4F07778 > 199900 )
  {
    if ( (*(_BYTE *)(a1 + 64) & 0x10) != 0 )
    {
      sub_6851C0(1804, a5);
      *a4 = 0;
    }
    else if ( *a4 )
    {
      sub_6851C0(404, a5);
      *a4 = 0;
    }
  }
  if ( (*(_QWORD *)(a3 + 8) & 0x180000) != 0 )
  {
    sub_6851C0((*(_QWORD *)(a3 + 8) & 0x100000LL) == 0 ? 2405 : 2932, a5);
    *(_QWORD *)(a3 + 8) &= 0xFFFFFFFFFFE7FFFFLL;
  }
  while ( *(_BYTE *)(v5 + 140) == 12 )
    v5 = *(_QWORD *)(v5 + 160);
  result = (__int64)&dword_4F077C4;
  v15 = *(__int64 **)(v5 + 168);
  if ( dword_4F077C4 == 2 )
  {
    if ( *((char *)v15 + 17) < 0 )
    {
      sub_684AA0(unk_4F07471, 2949, a5);
      v21 = *((_BYTE *)v15 + 17) & 0x7F;
      *((_BYTE *)v15 + 17) = v21;
    }
    else
    {
      v21 = *((unsigned __int8 *)v15 + 17);
    }
    result = v21 & 0xFFFFFF8F | 0x20;
    v22 = v15[7] == 0;
    *((_BYTE *)v15 + 17) = result;
    if ( !v22 )
    {
      result = sub_684B30(552, a1 + 24);
      v15[7] = 0;
    }
    if ( *(_BYTE *)(a3 + 269) == 2 )
    {
      result = sub_6851C0(378, a5);
      *(_BYTE *)(a3 + 269) = 0;
    }
  }
  v16 = *v15;
  if ( !*v15 )
    return result;
  v17 = *(_QWORD *)(v16 + 8);
  if ( v17 != v11 && !(unsigned int)sub_8D97D0(v17, v11, 0, v9, v10) )
  {
    sub_685310(1816, a5, *(_QWORD *)(v16 + 16));
    v18 = *(_QWORD **)v16;
    if ( v18 )
      goto LABEL_16;
    return sub_684B00(1817, a5);
  }
  v18 = *(_QWORD **)v16;
  if ( !v18 )
    return sub_684B00(1817, a5);
LABEL_16:
  if ( !(unsigned int)sub_8D2E30(v18[1])
    || (v19 = sub_8D46C0(v18[1]), !(unsigned int)sub_8D2E30(v19))
    || (v23 = sub_72BA30(unk_4F068B0), result = sub_8D46C0(v19), v23 != result)
    && (result = sub_8D97D0(result, v23, 0, v24, v25), !(_DWORD)result) )
  {
    result = sub_685310(1818, a5, v18[2]);
  }
  if ( *v18 )
    return sub_684B00(1817, a5);
  return result;
}
