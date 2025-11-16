// Function: sub_13C9440
// Address: 0x13c9440
//
__int64 __fastcall sub_13C9440(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  _QWORD *v4; // rax
  __int64 v5; // rax
  unsigned __int64 *v6; // rcx
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // rdx

  v2 = a1[6];
  v3 = a1[3];
  v4 = *(_QWORD **)(v2 + 48);
  if ( *(_QWORD **)(v2 + 56) == v4 )
  {
    v14 = &v4[*(unsigned int *)(v2 + 68)];
    if ( v4 == v14 )
    {
LABEL_21:
      v4 = v14;
    }
    else
    {
      while ( v3 != *v4 )
      {
        if ( v14 == ++v4 )
          goto LABEL_21;
      }
    }
  }
  else
  {
    v4 = (_QWORD *)sub_16CC9F0(v2 + 40, v3);
    if ( v3 == *v4 )
    {
      v12 = *(_QWORD *)(v2 + 56);
      if ( v12 == *(_QWORD *)(v2 + 48) )
        v13 = *(unsigned int *)(v2 + 68);
      else
        v13 = *(unsigned int *)(v2 + 64);
      v14 = (_QWORD *)(v12 + 8 * v13);
    }
    else
    {
      v5 = *(_QWORD *)(v2 + 56);
      if ( v5 != *(_QWORD *)(v2 + 48) )
        goto LABEL_4;
      v4 = (_QWORD *)(v5 + 8LL * *(unsigned int *)(v2 + 68));
      v14 = v4;
    }
  }
  if ( v4 != v14 )
  {
    *v4 = -2;
    ++*(_DWORD *)(v2 + 72);
  }
LABEL_4:
  v6 = (unsigned __int64 *)a1[5];
  v7 = a1[4] & 0xFFFFFFFFFFFFFFF8LL;
  *v6 = v7 | *v6 & 7;
  *(_QWORD *)(v7 + 8) = v6;
  v8 = a1[12];
  a1[4] &= 7uLL;
  a1[5] = 0;
  *a1 = &unk_49EA628;
  if ( v8 != a1[11] )
    _libc_free(v8);
  v9 = a1[9];
  if ( v9 != -8 && v9 != 0 && v9 != -16 )
    sub_1649B30(a1 + 7);
  *a1 = &unk_49EE2B0;
  v10 = a1[3];
  if ( v10 != 0 && v10 != -8 && v10 != -16 )
    sub_1649B30(a1 + 1);
  return j_j___libc_free_0(a1, 136);
}
