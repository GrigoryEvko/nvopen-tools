// Function: sub_163A340
// Address: 0x163a340
//
__int64 __fastcall sub_163A340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // r13

  v6 = a1;
  v7 = a2;
  if ( (unsigned __int8)sub_16D5D40(a1, a2, a3, a4, a5, a6) )
    sub_16C9040(a1);
  else
    ++*(_DWORD *)(a1 + 8);
  v12 = *(unsigned int *)(a1 + 40);
  if ( !(_DWORD)v12 )
    goto LABEL_10;
  a1 = (unsigned int)(v12 - 1);
  a2 = *(_QWORD *)(v6 + 24);
  v9 = (unsigned int)a1 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v8 = a2 + 16 * v9;
  v10 = *(_QWORD *)v8;
  if ( v7 != *(_QWORD *)v8 )
  {
    v8 = 1;
    while ( v10 != -4 )
    {
      v11 = (unsigned int)(v8 + 1);
      v9 = (unsigned int)a1 & ((_DWORD)v8 + (_DWORD)v9);
      v8 = a2 + 16LL * (unsigned int)v9;
      v10 = *(_QWORD *)v8;
      if ( v7 == *(_QWORD *)v8 )
        goto LABEL_5;
      v8 = (unsigned int)v11;
    }
LABEL_10:
    v13 = 0;
    if ( !(unsigned __int8)sub_16D5D40(a1, a2, v8, v9, v10, v11) )
      goto LABEL_7;
    goto LABEL_11;
  }
LABEL_5:
  if ( v8 == a2 + 16 * v12 )
    goto LABEL_10;
  v13 = *(_QWORD *)(v8 + 8);
  if ( !(unsigned __int8)sub_16D5D40(a1, a2, v8, v9, v10, v11) )
  {
LABEL_7:
    --*(_DWORD *)(v6 + 8);
    return v13;
  }
LABEL_11:
  sub_16C9060(v6);
  return v13;
}
