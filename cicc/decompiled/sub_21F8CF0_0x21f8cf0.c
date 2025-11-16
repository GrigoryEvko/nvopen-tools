// Function: sub_21F8CF0
// Address: 0x21f8cf0
//
__int64 __fastcall sub_21F8CF0(__int64 a1, int a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  int v5; // r8d
  unsigned int v6; // edx
  int *v7; // r13
  int v8; // esi
  unsigned __int64 *v9; // r14
  __int64 result; // rax
  __int64 v11; // rcx
  int v12; // r8d
  unsigned int v13; // edx
  int *v14; // r13
  int v15; // esi
  __int64 v16; // r12

  v3 = *(unsigned int *)(a1 + 256);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 240);
    v5 = 1;
    v6 = (v3 - 1) & (37 * a2);
    v7 = (int *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( v7 != (int *)(v4 + 16 * v3) )
      {
        v9 = (unsigned __int64 *)*((_QWORD *)v7 + 1);
        if ( v9 )
        {
          if ( (unsigned __int64 *)*v9 != v9 + 2 )
            _libc_free(*v9);
          j_j___libc_free_0(v9, 528);
        }
        *v7 = 0x80000000;
        --*(_DWORD *)(a1 + 248);
        ++*(_DWORD *)(a1 + 252);
      }
    }
    else
    {
      while ( v8 != 0x7FFFFFFF )
      {
        v6 = (v3 - 1) & (v5 + v6);
        v7 = (int *)(v4 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        ++v5;
      }
    }
  }
  result = *(unsigned int *)(a1 + 384);
  if ( (_DWORD)result )
  {
    v11 = *(_QWORD *)(a1 + 368);
    v12 = 1;
    v13 = (result - 1) & (37 * a2);
    v14 = (int *)(v11 + 16LL * v13);
    v15 = *v14;
    if ( *v14 == a2 )
    {
LABEL_11:
      result = v11 + 16 * result;
      if ( v14 != (int *)result )
      {
        v16 = *((_QWORD *)v14 + 1);
        if ( v16 )
        {
          j___libc_free_0(*(_QWORD *)(v16 + 8));
          result = j_j___libc_free_0(v16, 32);
        }
        *v14 = 0x80000000;
        --*(_DWORD *)(a1 + 376);
        ++*(_DWORD *)(a1 + 380);
      }
    }
    else
    {
      while ( v15 != 0x7FFFFFFF )
      {
        v13 = (result - 1) & (v12 + v13);
        v14 = (int *)(v11 + 16LL * v13);
        v15 = *v14;
        if ( *v14 == a2 )
          goto LABEL_11;
        ++v12;
      }
    }
  }
  return result;
}
