// Function: sub_185C3B0
// Address: 0x185c3b0
//
__int64 __fastcall sub_185C3B0(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r13
  _QWORD *v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // rdx

  sub_159D9E0(a1);
  v2 = *(_BYTE *)(a1 + 32) & 0xF;
  if ( ((v2 + 9) & 0xFu) > 1 && ((v2 + 15) & 0xFu) > 2 && !sub_15E4F60(a1) )
    return 0;
  v3 = sub_15E4F10(a1);
  if ( !v3 || (*(_BYTE *)(a1 + 32) & 0xFu) - 7 <= 1 )
    goto LABEL_5;
  v5 = *(_QWORD **)(a2 + 16);
  v6 = *(_QWORD **)(a2 + 8);
  if ( v5 == v6 )
  {
    v7 = &v6[*(unsigned int *)(a2 + 28)];
    if ( v6 == v7 )
    {
      v11 = *(_QWORD **)(a2 + 8);
    }
    else
    {
      do
      {
        if ( v3 == *v6 )
          break;
        ++v6;
      }
      while ( v7 != v6 );
      v11 = v7;
    }
  }
  else
  {
    v7 = &v5[*(unsigned int *)(a2 + 24)];
    v6 = sub_16CC9F0(a2, v3);
    if ( v3 == *v6 )
    {
      v9 = *(_QWORD *)(a2 + 16);
      if ( v9 == *(_QWORD *)(a2 + 8) )
        v10 = *(unsigned int *)(a2 + 28);
      else
        v10 = *(unsigned int *)(a2 + 24);
      v11 = (_QWORD *)(v9 + 8 * v10);
    }
    else
    {
      v8 = *(_QWORD *)(a2 + 16);
      if ( v8 != *(_QWORD *)(a2 + 8) )
      {
        v6 = (_QWORD *)(v8 + 8LL * *(unsigned int *)(a2 + 24));
        goto LABEL_16;
      }
      v6 = (_QWORD *)(v8 + 8LL * *(unsigned int *)(a2 + 28));
      v11 = v6;
    }
  }
  while ( v11 != v6 && *v6 >= 0xFFFFFFFFFFFFFFFELL )
    ++v6;
LABEL_16:
  if ( v6 == v7 )
  {
LABEL_5:
    if ( *(_BYTE *)(a1 + 16) )
    {
      if ( *(_QWORD *)(a1 + 8) )
        return 0;
    }
    else if ( (!sub_15E4F60(a1) || *(_QWORD *)(a1 + 8)) && !(unsigned __int8)sub_15E36F0(a1) )
    {
      return 0;
    }
    sub_15E5B20(a1);
    return 1;
  }
  return 0;
}
