// Function: sub_D110B0
// Address: 0xd110b0
//
__int64 __fastcall sub_D110B0(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // r14
  _QWORD *v4; // r12
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rax
  _QWORD *v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  _BOOL8 v14; // rdi
  _QWORD *v15; // r14
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rax
  _QWORD *v19; // rdi

  v2 = a1 + 2;
  v4 = a1 + 2;
  v6 = (_QWORD *)a1[3];
  if ( !v6 )
    goto LABEL_10;
  do
  {
    while ( 1 )
    {
      v7 = v6[2];
      v8 = v6[3];
      if ( v6[4] >= a2 )
        break;
      v6 = (_QWORD *)v6[3];
      if ( !v8 )
        goto LABEL_6;
    }
    v4 = v6;
    v6 = (_QWORD *)v6[2];
  }
  while ( v7 );
LABEL_6:
  if ( v2 == v4 || v4[4] > a2 )
  {
LABEL_10:
    v10 = sub_22077B0(48);
    v11 = v4;
    *(_QWORD *)(v10 + 32) = a2;
    v4 = (_QWORD *)v10;
    *(_QWORD *)(v10 + 40) = 0;
    v12 = sub_D10FB0(a1 + 1, v11, (unsigned __int64 *)(v10 + 32));
    if ( v13 )
    {
      v14 = v2 == v13 || v12 || a2 < v13[4];
      sub_220F040(v14, v4, v13, v2);
      ++a1[6];
      result = v4[5];
      if ( result )
        return result;
      goto LABEL_15;
    }
    v19 = v4;
    v4 = v12;
    j_j___libc_free_0(v19, 48);
  }
  result = v4[5];
  if ( result )
    return result;
LABEL_15:
  result = sub_22077B0(48);
  if ( result )
  {
    *(_QWORD *)result = a1;
    *(_QWORD *)(result + 8) = a2;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_QWORD *)(result + 32) = 0;
    *(_DWORD *)(result + 40) = 0;
  }
  v15 = (_QWORD *)v4[5];
  v4[5] = result;
  if ( v15 )
  {
    v16 = v15[3];
    v17 = v15[2];
    if ( v16 != v17 )
    {
      do
      {
        if ( *(_BYTE *)(v17 + 24) )
        {
          v18 = *(_QWORD *)(v17 + 16);
          *(_BYTE *)(v17 + 24) = 0;
          if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
            sub_BD60C0((_QWORD *)v17);
        }
        v17 += 40;
      }
      while ( v16 != v17 );
      v17 = v15[2];
    }
    if ( v17 )
      j_j___libc_free_0(v17, v15[4] - v17);
    j_j___libc_free_0(v15, 48);
    return v4[5];
  }
  return result;
}
