// Function: sub_2EABED0
// Address: 0x2eabed0
//
__int64 __fastcall sub_2EABED0(__int64 a1, unsigned int a2, char a3, unsigned __int8 *a4, size_t a5)
{
  __int64 v6; // r12
  _DWORD *v7; // rdx
  unsigned __int64 v8; // rax
  __int64 result; // rax
  _BYTE *v12; // rax
  void *v13; // rdi
  __int64 v14; // rax

  v6 = a1;
  v7 = *(_DWORD **)(a1 + 32);
  v8 = *(_QWORD *)(a1 + 24) - (_QWORD)v7;
  if ( a3 )
  {
    if ( v8 <= 0xC )
    {
      v6 = sub_CB6200(a1, "%fixed-stack.", 0xDu);
    }
    else
    {
      qmemcpy(v7, "%fixed-stack.", 13);
      *(_QWORD *)(a1 + 32) += 13LL;
    }
    return sub_CB59D0(v6, a2);
  }
  else
  {
    if ( v8 <= 6 )
    {
      v14 = sub_CB6200(a1, "%stack.", 7u);
      result = sub_CB59D0(v14, a2);
      if ( !a5 )
        return result;
    }
    else
    {
      *v7 = 1635021605;
      *((_WORD *)v7 + 2) = 27491;
      *((_BYTE *)v7 + 6) = 46;
      *(_QWORD *)(a1 + 32) += 7LL;
      result = sub_CB59D0(a1, a2);
      if ( !a5 )
        return result;
    }
    v12 = *(_BYTE **)(a1 + 32);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(a1 + 24) )
    {
      v6 = sub_CB5D20(a1, 46);
    }
    else
    {
      *(_QWORD *)(a1 + 32) = v12 + 1;
      *v12 = 46;
    }
    v13 = *(void **)(v6 + 32);
    if ( *(_QWORD *)(v6 + 24) - (_QWORD)v13 < a5 )
    {
      return sub_CB6200(v6, a4, a5);
    }
    else
    {
      result = (__int64)memcpy(v13, a4, a5);
      *(_QWORD *)(v6 + 32) += a5;
    }
  }
  return result;
}
