// Function: sub_1E31F40
// Address: 0x1e31f40
//
__int64 __fastcall sub_1E31F40(__int64 a1, unsigned int a2, char a3, char *a4, size_t a5)
{
  __int64 v6; // r12
  _DWORD *v7; // rdx
  unsigned __int64 v8; // rax
  __int64 result; // rax
  _BYTE *v12; // rax
  void *v13; // rdi
  __int64 v14; // rax

  v6 = a1;
  v7 = *(_DWORD **)(a1 + 24);
  v8 = *(_QWORD *)(a1 + 16) - (_QWORD)v7;
  if ( a3 )
  {
    if ( v8 <= 0xC )
    {
      v6 = sub_16E7EE0(a1, "%fixed-stack.", 0xDu);
    }
    else
    {
      qmemcpy(v7, "%fixed-stack.", 13);
      *(_QWORD *)(a1 + 24) += 13LL;
    }
    return sub_16E7A90(v6, a2);
  }
  else
  {
    if ( v8 <= 6 )
    {
      v14 = sub_16E7EE0(a1, "%stack.", 7u);
      result = sub_16E7A90(v14, a2);
      if ( !a5 )
        return result;
    }
    else
    {
      *v7 = 1635021605;
      *((_WORD *)v7 + 2) = 27491;
      *((_BYTE *)v7 + 6) = 46;
      *(_QWORD *)(a1 + 24) += 7LL;
      result = sub_16E7A90(a1, a2);
      if ( !a5 )
        return result;
    }
    v12 = *(_BYTE **)(a1 + 24);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(a1 + 16) )
    {
      v6 = sub_16E7DE0(a1, 46);
    }
    else
    {
      *(_QWORD *)(a1 + 24) = v12 + 1;
      *v12 = 46;
    }
    v13 = *(void **)(v6 + 24);
    if ( *(_QWORD *)(v6 + 16) - (_QWORD)v13 < a5 )
    {
      return sub_16E7EE0(v6, a4, a5);
    }
    else
    {
      result = (__int64)memcpy(v13, a4, a5);
      *(_QWORD *)(v6 + 24) += a5;
    }
  }
  return result;
}
