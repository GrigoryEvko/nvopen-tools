// Function: sub_19C06B0
// Address: 0x19c06b0
//
__int64 __fastcall sub_19C06B0(__int64 a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rsi
  _QWORD *v4; // r8
  _QWORD *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  unsigned __int64 v13; // rdi

  v2 = *(_QWORD **)(a1 + 232);
  v3 = *(_QWORD *)(a1 + 296);
  if ( v2 )
  {
    v4 = (_QWORD *)(a1 + 224);
    v5 = (_QWORD *)(a1 + 224);
    do
    {
      while ( 1 )
      {
        v6 = v2[2];
        v7 = v2[3];
        if ( v2[4] >= v3 )
          break;
        v2 = (_QWORD *)v2[3];
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v2;
      v2 = (_QWORD *)v2[2];
    }
    while ( v6 );
LABEL_6:
    if ( v4 != v5 && v5[4] <= v3 )
    {
      *(_DWORD *)(a1 + 280) += *((_DWORD *)v5 + 12) * (*((_DWORD *)v5 + 10) + *((_DWORD *)v5 + 11));
      v9 = sub_220F330(v5, v4);
      v10 = *(unsigned int *)(v9 + 80);
      if ( (_DWORD)v10 )
      {
        v11 = *(_QWORD **)(v9 + 64);
        v12 = &v11[14 * v10];
        do
        {
          if ( *v11 != -16 && *v11 != -8 )
          {
            v13 = v11[3];
            if ( v13 != v11[2] )
              _libc_free(v13);
          }
          v11 += 14;
        }
        while ( v12 != v11 );
      }
      j___libc_free_0(*(_QWORD *)(v9 + 64));
      j_j___libc_free_0(v9, 88);
      --*(_QWORD *)(a1 + 256);
    }
  }
  result = (unsigned int)dword_4FB3640;
  if ( *(_DWORD *)(a1 + 280) > (unsigned int)dword_4FB3640 )
    *(_DWORD *)(a1 + 280) = dword_4FB3640;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  return result;
}
