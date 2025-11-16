// Function: sub_193E640
// Address: 0x193e640
//
__int64 __fastcall sub_193E640(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rax
  int v6; // esi
  char v8; // di
  unsigned int v9; // esi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // r13
  _QWORD *v16; // rax
  _QWORD *v18; // rax

  v3 = 0x17FFFFFFE8LL;
  v6 = *(_DWORD *)(a1 + 20);
  v8 = *(_BYTE *)(a1 + 23) & 0x40;
  v9 = v6 & 0xFFFFFFF;
  if ( v9 )
  {
    v10 = 24LL * *(unsigned int *)(a1 + 56) + 8;
    v11 = 0;
    do
    {
      v12 = a1 - 24LL * v9;
      if ( v8 )
        v12 = *(_QWORD *)(a1 - 8);
      if ( a2 == *(_QWORD *)(v12 + v10) )
      {
        v3 = 24 * v11;
        goto LABEL_8;
      }
      ++v11;
      v10 += 8;
    }
    while ( v9 != (_DWORD)v11 );
    v3 = 0x17FFFFFFE8LL;
  }
LABEL_8:
  if ( v8 )
    v13 = *(_QWORD *)(a1 - 8);
  else
    v13 = a1 - 24LL * v9;
  v14 = *(_QWORD *)(a1 + 8);
  v15 = *(_QWORD *)(v13 + v3);
  if ( v14 )
  {
    while ( 1 )
    {
      v16 = sub_1648700(v14);
      if ( a3 != v16 && v16 != (_QWORD *)v15 )
        return 0;
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
        goto LABEL_16;
    }
  }
  else
  {
LABEL_16:
    while ( 1 )
    {
      v15 = *(_QWORD *)(v15 + 8);
      if ( !v15 )
        break;
      while ( 1 )
      {
        v18 = sub_1648700(v15);
        if ( a3 == v18 )
          break;
        if ( (_QWORD *)a1 != v18 )
          return 0;
        v15 = *(_QWORD *)(v15 + 8);
        if ( !v15 )
          return 1;
      }
    }
    return 1;
  }
}
