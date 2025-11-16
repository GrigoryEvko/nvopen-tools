// Function: sub_163A430
// Address: 0x163a430
//
__int64 __fastcall sub_163A430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r13

  if ( (unsigned __int8)sub_16D5D40(a1, a2, a3, a4, a5, a6) )
    sub_16C9040(a1);
  else
    ++*(_DWORD *)(a1 + 8);
  v8 = a1 + 48;
  v9 = sub_16D1B30(a1 + 48, a2, a3);
  if ( v9 == -1
    || (v14 = *(_QWORD *)(a1 + 48),
        v11 = *(unsigned int *)(a1 + 56),
        v15 = v14 + 8LL * v9,
        v10 = v14 + 8 * v11,
        v15 == v10) )
  {
    v16 = 0;
    if ( !(unsigned __int8)sub_16D5D40(v8, a2, v10, v11, v12, v13) )
      goto LABEL_6;
  }
  else
  {
    v16 = *(_QWORD *)(*(_QWORD *)v15 + 8LL);
    if ( !(unsigned __int8)sub_16D5D40(v8, a2, v10, v11, v12, v13) )
    {
LABEL_6:
      --*(_DWORD *)(a1 + 8);
      return v16;
    }
  }
  sub_16C9060(a1);
  return v16;
}
