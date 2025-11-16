// Function: sub_731F80
// Address: 0x731f80
//
__int64 __fastcall sub_731F80(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, int *a5, int *a6)
{
  int v6; // r14d
  char v10; // dl
  __int64 v11; // rax
  unsigned int v12; // r15d
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  char v17; // di
  int v18; // r14d
  _DWORD *v19; // rdx
  int v20; // eax
  int v21; // r13d
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax

  v6 = a2;
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a1) )
    sub_8AE000(a1);
  v10 = *(_BYTE *)(a1 + 140);
  if ( v10 == 12 )
  {
    v11 = a1;
    do
    {
      v11 = *(_QWORD *)(v11 + 160);
      v10 = *(_BYTE *)(v11 + 140);
    }
    while ( v10 == 12 );
  }
  if ( !v10 )
  {
    v21 = 0;
    v18 = 1;
    v12 = 0;
    goto LABEL_27;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( dword_4F04C44 != -1
      || (v23 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v23 + 6) & 6) != 0)
      || *(_BYTE *)(v23 + 4) == 12 )
    {
      if ( (unsigned int)sub_8DBE70(a1) )
      {
        v21 = 1;
        v18 = 0;
        v12 = 0;
        goto LABEL_27;
      }
    }
  }
  v12 = 0;
  if ( a3 )
  {
    if ( *(_BYTE *)(a3 + 24) == 1 && (unsigned __int8)(*(_BYTE *)(a3 + 56) - 94) <= 1u )
    {
      a2 = 1;
      v12 = sub_7A7D30(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 72) + 16LL) + 56LL), 1);
    }
    if ( (!HIDWORD(qword_4F077B4) || qword_4F077A8 > 0x7593u)
      && (*(_BYTE *)(a3 + 25) & 1) != 0
      && *(_BYTE *)(a3 + 24) == 3
      && *(_DWORD *)(*(_QWORD *)(a3 + 56) + 152LL) )
    {
      v12 = *(_DWORD *)(*(_QWORD *)(a3 + 56) + 152LL);
    }
  }
  v13 = sub_8D4130(a1);
  if ( !(unsigned int)sub_8D23B0(v13) )
  {
    v25 = sub_8D4130(a1);
    v18 = sub_8D2BE0(v25);
    if ( v18 )
    {
      v18 = 1;
      if ( a4 )
      {
        a2 = 3414;
        sub_6E5D20(8, 0xD56u, a4, a1);
      }
    }
    goto LABEL_17;
  }
  if ( !HIDWORD(qword_4F077B4) || !v6 || (unsigned int)sub_8D2600(a1) )
  {
    v17 = 5;
    v18 = dword_4D04964;
    if ( !dword_4D04964 )
    {
      v19 = a4;
      if ( !a4 )
        goto LABEL_17;
      goto LABEL_16;
    }
  }
  v19 = a4;
  v17 = 8;
  v18 = 1;
  if ( a4 )
  {
LABEL_16:
    a2 = 1273;
    sub_6E5C80(v17, 0x4F9u, v19);
  }
LABEL_17:
  if ( dword_4F04C44 == -1
    && (v19 = qword_4F04C68, v24 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v24 + 6) & 6) == 0)
    && *(_BYTE *)(v24 + 4) != 12
    || (!a3 ? (v20 = sub_8DC060(a1)) : (v20 = sub_731EE0(a3, a2, (__int64)v19, v14, v15, v16)), v21 = 1, !v20) )
  {
    v21 = 0;
    if ( !v18 && !v12 )
    {
      if ( *(char *)(a1 + 142) >= 0 && *(_BYTE *)(a1 + 140) == 12 )
      {
        v12 = sub_8D4AB0(a1, a2, v19);
      }
      else
      {
        v12 = *(_DWORD *)(a1 + 136);
        v21 = 0;
      }
    }
  }
LABEL_27:
  *a6 = v21;
  *a5 = v18;
  return v12;
}
