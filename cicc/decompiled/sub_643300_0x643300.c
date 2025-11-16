// Function: sub_643300
// Address: 0x643300
//
__int64 __fastcall sub_643300(__int64 a1, int *a2, _DWORD *a3)
{
  _QWORD *v5; // r15
  __int64 v6; // rdi
  bool v7; // zf
  __int64 v8; // rbx
  __int64 result; // rax
  int v10; // edx
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  int v14; // [rsp+8h] [rbp-38h] BYREF
  _DWORD v15[13]; // [rsp+Ch] [rbp-34h] BYREF

  v14 = 0;
  v15[0] = 0;
  if ( (*(_BYTE *)(a1 + 124) & 1) != 0 )
  {
    v5 = *(_QWORD **)(sub_735B70() + 128);
    v6 = v5[13];
    if ( v6 )
      goto LABEL_3;
  }
  else
  {
    v5 = *(_QWORD **)(a1 + 128);
    v6 = v5[13];
    if ( v6 )
    {
LABEL_3:
      v7 = *(_QWORD *)(a1 + 8) == 0;
      v14 = 1;
      if ( v7 )
      {
        result = 1;
        v10 = 1;
        goto LABEL_14;
      }
      v8 = v5[21];
      if ( v8 )
        goto LABEL_8;
      goto LABEL_22;
    }
  }
  if ( v5[14] || v5[18] || v5[19] )
    goto LABEL_3;
  v8 = v5[21];
  if ( !v8 )
  {
LABEL_12:
    v10 = v14;
    if ( !v14 )
    {
LABEL_13:
      result = v15[0];
      goto LABEL_14;
    }
    v6 = v5[13];
LABEL_22:
    if ( v6 && (unsigned int)sub_6431C0(v6, 6u)
      || (v11 = v5[14]) != 0 && (unsigned int)sub_6431C0(v11, 7u)
      || (v12 = v5[18]) != 0 && (unsigned int)sub_6431C0(v12, 0xBu)
      || (v13 = v5[19]) != 0 && (unsigned int)sub_6431C0(v13, 0x2Bu) )
    {
      v10 = v14;
      result = 1;
      goto LABEL_14;
    }
    v10 = v14;
    goto LABEL_13;
  }
  while ( 1 )
  {
LABEL_8:
    while ( 1 )
    {
      if ( *(_QWORD *)(v8 + 8) )
      {
        v14 = 1;
        if ( (*(_BYTE *)(v8 + 124) & 2) == 0 )
          break;
      }
      sub_643300(v8, &v14, v15);
      if ( v14 )
        break;
      v8 = *(_QWORD *)(v8 + 112);
      if ( !v8 )
        goto LABEL_12;
    }
    result = v15[0];
    if ( v15[0] )
      break;
    v8 = *(_QWORD *)(v8 + 112);
    if ( !v8 )
      goto LABEL_12;
  }
  v10 = v14;
LABEL_14:
  *a2 = v10;
  *a3 = result;
  return result;
}
