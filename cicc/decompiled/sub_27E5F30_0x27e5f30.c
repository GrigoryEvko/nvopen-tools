// Function: sub_27E5F30
// Address: 0x27e5f30
//
__int64 __fastcall sub_27E5F30(__int64 a1, _QWORD *a2, __int64 ***a3, __int64 a4)
{
  _QWORD *v6; // rdx
  _QWORD *v7; // rcx
  _QWORD *v8; // rax
  unsigned __int64 v10; // rax
  int v11; // edx
  unsigned __int8 *v12; // rax
  bool v13; // cf
  unsigned __int8 *v14; // rdx

  if ( (_QWORD *)a4 == a2 )
    return 0;
  if ( *(_BYTE *)(a1 + 124) )
  {
    v6 = *(_QWORD **)(a1 + 104);
    v7 = &v6[*(unsigned int *)(a1 + 116)];
    if ( v6 != v7 )
    {
      v8 = *(_QWORD **)(a1 + 104);
      while ( a2 != (_QWORD *)*v8 )
      {
        if ( v7 == ++v8 )
          goto LABEL_10;
      }
      return 0;
    }
LABEL_15:
    v10 = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v10 == a2 + 6 )
    {
      v14 = 0;
    }
    else
    {
      if ( !v10 )
        BUG();
      v11 = *(unsigned __int8 *)(v10 - 24);
      v12 = (unsigned __int8 *)(v10 - 24);
      v13 = (unsigned int)(v11 - 30) < 0xB;
      v14 = 0;
      if ( v13 )
        v14 = v12;
    }
    if ( (unsigned int)sub_27DC180(*(__int64 ***)(a1 + 24), a2, v14, *(_DWORD *)(a1 + 416)) <= *(_DWORD *)(a1 + 416) )
    {
      sub_27E59E0((__int64 *)a1, (unsigned __int64)a2, a3, a4);
      return 1;
    }
    return 0;
  }
  if ( !sub_C8CA60(a1 + 96, (__int64)a2) )
  {
    if ( *(_BYTE *)(a1 + 124) )
    {
      v6 = *(_QWORD **)(a1 + 104);
      v8 = &v6[*(unsigned int *)(a1 + 116)];
      if ( v6 != v8 )
      {
LABEL_10:
        while ( a4 != *v6 )
        {
          if ( ++v6 == v8 )
            goto LABEL_15;
        }
        return 0;
      }
      goto LABEL_15;
    }
    if ( !sub_C8CA60(a1 + 96, a4) )
      goto LABEL_15;
  }
  return 0;
}
