// Function: sub_2361BD0
// Address: 0x2361bd0
//
void __fastcall sub_2361BD0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned __int64 v17; // rsi

  if ( a1 != a2 )
  {
    v7 = *a2;
    v8 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 && ((v9 = v7 >> 2, (v7 & 4) == 0) || *(_DWORD *)(v8 + 8)) )
    {
      v10 = *a1;
      if ( !*a1 || (v10 & 4) == 0 || (v11 = v10 & 0xFFFFFFFFFFFFFFF8LL, (v12 = v11) == 0) )
      {
LABEL_12:
        *a1 = *a2;
        *a2 = 0;
        return;
      }
      v13 = v9 & 1;
      if ( (_DWORD)v13 )
      {
        if ( *(_QWORD *)v11 != v11 + 16 )
          _libc_free(*(_QWORD *)v11);
        j_j___libc_free_0(v12);
        goto LABEL_12;
      }
      *(_DWORD *)(v11 + 8) = 0;
      v16 = *a2;
      if ( (*a2 & 4) != 0 )
        v17 = **(_QWORD **)(v16 & 0xFFFFFFFFFFFFFFF8LL);
      else
        v17 = v16 & 0xFFFFFFFFFFFFFFF8LL;
      sub_2361B80(v11, v17, v8, v13, a5, a6);
      *a2 = 0;
    }
    else
    {
      v14 = *a1;
      if ( ((*a1 >> 2) & 1) != 0 )
      {
        if ( v14 )
        {
          if ( ((*a1 >> 2) & 1) != 0 )
          {
            v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v15 )
              *(_DWORD *)(v15 + 8) = 0;
          }
        }
      }
      else
      {
        *a1 = 0;
      }
    }
  }
}
