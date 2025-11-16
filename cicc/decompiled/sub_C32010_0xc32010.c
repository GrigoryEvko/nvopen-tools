// Function: sub_C32010
// Address: 0xc32010
//
__int64 __fastcall sub_C32010(__int64 a1, _BYTE *a2, _BYTE *a3, _QWORD *a4)
{
  _BYTE *v7; // rax
  char v8; // dl
  _BYTE *v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r13
  unsigned int v14; // r14d
  __int64 v15; // rax
  __int64 v16; // rbx
  const char *v17; // [rsp+0h] [rbp-50h] BYREF
  char v18; // [rsp+20h] [rbp-30h]
  char v19; // [rsp+21h] [rbp-2Fh]

  *a4 = a3;
  if ( a2 == a3 )
  {
LABEL_12:
    v7 = a3;
LABEL_7:
    v8 = *(_BYTE *)(a1 + 8);
    *(_QWORD *)a1 = v7;
    *(_BYTE *)(a1 + 8) = v8 & 0xFC | 2;
    return a1;
  }
  v7 = a2;
  while ( *v7 == 48 )
  {
    if ( ++v7 == a3 )
      goto LABEL_12;
  }
  if ( a3 == v7 || *v7 != 46 )
    goto LABEL_7;
  *a4 = v7;
  v10 = v7 + 1;
  if ( a3 - a2 != 1 )
  {
    while ( v10 != a3 )
    {
      if ( *v10 != 48 )
      {
        v7 = v10;
        goto LABEL_7;
      }
      ++v10;
    }
    goto LABEL_12;
  }
  v19 = 1;
  v17 = "Significand has no digits";
  v18 = 3;
  v11 = sub_C63BB0(a2, a3, v10, a4);
  v13 = v12;
  v14 = v11;
  v15 = sub_22077B0(64);
  v16 = v15;
  if ( v15 )
    sub_C63EB0(v15, &v17, v14, v13);
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v16 & 0xFFFFFFFFFFFFFFFELL;
  return a1;
}
