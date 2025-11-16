// Function: sub_386B550
// Address: 0x386b550
//
unsigned __int64 *__fastcall sub_386B550(__int64 *a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r14
  __int64 *v4; // rbx
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rax
  int v9; // edx
  __int64 v10; // rcx
  __int64 v11; // r8

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 23 )
  {
    v3 = sub_386AFE0((__int64 *)a2);
  }
  else
  {
    v3 = *(_QWORD *)(a2 - 24);
    if ( v2 == 21 )
      goto LABEL_21;
  }
  v4 = *(__int64 **)(a2 + 8);
  if ( v4 )
  {
    if ( (*(_BYTE *)(a2 + 17) & 1) != 0 )
    {
      sub_164CC90(a2, v3);
      v4 = *(__int64 **)(a2 + 8);
    }
    while ( v4 )
    {
      v8 = sub_1648700((__int64)v4);
      v9 = *((unsigned __int8 *)v8 + 16);
      if ( v9 == 21 )
      {
        *((_DWORD *)v8 + 21) = -1;
      }
      else if ( v9 == 22 )
      {
        *((_DWORD *)v8 + 22) = -1;
      }
      if ( *v4 )
      {
        v5 = v4[1];
        v6 = v4[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v6 = v5;
        if ( v5 )
          *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
      }
      *v4 = v3;
      if ( v3 )
      {
        v7 = *(_QWORD *)(v3 + 8);
        v4[1] = v7;
        if ( v7 )
          *(_QWORD *)(v7 + 16) = (unsigned __int64)(v4 + 1) | *(_QWORD *)(v7 + 16) & 3LL;
        v4[2] = (v3 + 8) | v4[2] & 3;
        *(_QWORD *)(v3 + 8) = v4;
      }
      v4 = *(__int64 **)(a2 + 8);
    }
  }
LABEL_21:
  sub_14222A0(*a1, a2);
  return sub_1422460(*a1, a2, 1, v10, v11);
}
