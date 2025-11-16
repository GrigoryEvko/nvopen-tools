// Function: sub_2EC0780
// Address: 0x2ec0780
//
__int64 __fastcall sub_2EC0780(__int64 a1, int a2, _BYTE *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // r15d
  __int64 v11; // rsi
  _BOOL8 v12; // rcx
  bool v13; // al
  unsigned __int8 v14; // di
  __int64 **v15; // rax
  __int64 v16; // rdx
  __int64 **v17; // r14
  __int64 *v18; // rdi
  __int64 **v19; // r13
  __int64 v21; // rax
  __int64 (__fastcall *v22)(__int64); // rcx
  __int64 **v23; // rax
  _BYTE *v24; // rdx

  v10 = sub_2EC0130(a1, a3, a4, a4, a5, a6);
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (v10 & 0x7FFFFFFF)) = *(_QWORD *)(*(_QWORD *)(a1 + 56)
                                                                            + 16LL * (a2 & 0x7FFFFFFF));
  if ( a2 >= 0 || (a2 & 0x7FFFFFFFu) >= *(_DWORD *)(a1 + 464) )
  {
    v11 = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
  }
  else
  {
    v24 = (_BYTE *)(*(_QWORD *)(a1 + 456) + 8LL * (a2 & 0x7FFFFFFF));
    v14 = *v24 & 1;
    v12 = (*v24 & 4) != 0;
    v11 = *(_QWORD *)v24 >> 3;
    v13 = (*v24 & 2) != 0;
  }
  sub_2EBE740(a1, v10, (8 * v11) | (4 * v12) | v14 | (2LL * v13), v12, v8, v9);
  v15 = *(__int64 ***)(a1 + 16);
  if ( *(_BYTE *)(a1 + 36) )
    v16 = *(unsigned int *)(a1 + 28);
  else
    v16 = *(unsigned int *)(a1 + 24);
  v17 = &v15[v16];
  if ( v15 != v17 )
  {
    while ( 1 )
    {
      v18 = *v15;
      v19 = v15;
      if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v17 == ++v15 )
        return v10;
    }
    if ( v17 != v15 )
    {
      v21 = *v18;
      v22 = *(__int64 (__fastcall **)(__int64))(*v18 + 32);
      if ( v22 != sub_2EBDF90 )
        goto LABEL_18;
LABEL_11:
      (*(void (__fastcall **)(__int64 *, _QWORD))(v21 + 24))(v18, v10);
      while ( 1 )
      {
        v23 = v19 + 1;
        if ( v19 + 1 == v17 )
          break;
        v18 = *v23;
        for ( ++v19; (unsigned __int64)*v23 >= 0xFFFFFFFFFFFFFFFELL; v19 = v23 )
        {
          if ( v17 == ++v23 )
            return v10;
          v18 = *v23;
        }
        if ( v17 == v19 )
          return v10;
        v21 = *v18;
        v22 = *(__int64 (__fastcall **)(__int64))(*v18 + 32);
        if ( v22 == sub_2EBDF90 )
          goto LABEL_11;
LABEL_18:
        ((void (__fastcall *)(__int64 *, _QWORD, _QWORD))v22)(v18, v10, (unsigned int)a2);
      }
    }
  }
  return v10;
}
