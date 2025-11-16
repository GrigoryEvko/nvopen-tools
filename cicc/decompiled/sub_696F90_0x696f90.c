// Function: sub_696F90
// Address: 0x696f90
//
__int64 __fastcall sub_696F90(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, _DWORD *a5, __int64 a6, __int64 a7)
{
  __int64 *v10; // rbx
  __int64 v11; // rax
  char v12; // dl
  __int64 i; // r15
  __int64 result; // rax
  int v15; // eax
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned int v19; // [rsp+1Ch] [rbp-1A4h] BYREF
  __int64 v20; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v21; // [rsp+28h] [rbp-198h] BYREF
  _QWORD v22[2]; // [rsp+30h] [rbp-190h] BYREF
  char v23; // [rsp+41h] [rbp-17Fh]
  char v24; // [rsp+42h] [rbp-17Eh]

  v10 = (__int64 *)(a3 + 8);
  v20 = 0;
  v21 = 0;
  v19 = 0;
  if ( a2 )
  {
    v10 = v22;
    sub_6E6A50(a2, v22);
    if ( *(_BYTE *)(a2 + 173) == 6 && v23 == 2 )
    {
      if ( (unsigned int)sub_8D32E0(v22[0]) )
      {
        v22[0] = sub_8D46C0(v22[0]);
        sub_6E6A20(v22);
        v17 = *(_QWORD *)(a2 + 144);
        if ( v17 )
        {
          if ( *(_BYTE *)(v17 + 24) == 3 )
            v24 = v24 & 0xD7 | *(_BYTE *)(v17 + 26) & 0x20 | 8;
        }
      }
    }
  }
  v11 = sub_8D4940(a1);
  v12 = *(_BYTE *)(v11 + 140);
  for ( i = v11; v12 == 12; v12 = *(_BYTE *)(v11 + 140) )
    v11 = *(_QWORD *)(v11 + 160);
  if ( !v12 )
  {
LABEL_6:
    if ( a4 )
    {
LABEL_7:
      result = 1;
LABEL_8:
      *a4 = i;
      return result;
    }
    return 1;
  }
  v15 = sub_8D3F60(i);
  v16 = (*(_BYTE *)(i + 161) & 8) != 0;
  if ( (unsigned int)sub_696CB0(v16, v15, 1u, 0, a1, (_BYTE *)i, 0, (__int64)v10, 0, (__int64)a5, &v20, &v21, &v19) )
  {
    if ( !a7
      || *(_DWORD *)(*(_QWORD *)(i + 168) + 28LL) == -2
      || (v16 = i, (unsigned int)sub_89FB00(i, v21, a6, a7, 0, a5)) )
    {
      v16 = (__int64)&v20;
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 0x10) != 0 )
        a5 = 0;
      if ( (unsigned int)sub_8AE140(&v20, 1, a5) )
      {
        if ( a4 )
        {
          i = v20;
          goto LABEL_7;
        }
        return 1;
      }
    }
  }
  else
  {
    if ( v19 )
    {
      i = a1;
      goto LABEL_6;
    }
    if ( a5 )
    {
      v16 = 2886;
      sub_6861A0(0xB46u, a5, a1, *v10);
    }
  }
  if ( a4 )
  {
    i = sub_72C930(v16);
    result = 0;
    goto LABEL_8;
  }
  return 0;
}
