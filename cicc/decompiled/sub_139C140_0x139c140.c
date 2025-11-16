// Function: sub_139C140
// Address: 0x139c140
//
void __fastcall sub_139C140(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  char v4; // dl
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdi
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  unsigned int v10; // r8d
  _QWORD *v11; // rcx

  if ( a2 )
  {
    v3 = a2;
    while ( 1 )
    {
      v7 = *a1;
      v8 = *(_QWORD **)(*a1 + 8);
      if ( *(_QWORD **)(*a1 + 16) == v8 )
      {
        v9 = &v8[*(unsigned int *)(v7 + 28)];
        v10 = *(_DWORD *)(v7 + 28);
        if ( v8 != v9 )
        {
          v11 = 0;
          while ( *v8 != v3 )
          {
            if ( *v8 == -2 )
              v11 = v8;
            if ( v9 == ++v8 )
            {
              if ( !v11 )
                goto LABEL_19;
              *v11 = v3;
              --*(_DWORD *)(v7 + 32);
              ++*(_QWORD *)v7;
              goto LABEL_4;
            }
          }
          goto LABEL_8;
        }
LABEL_19:
        if ( v10 < *(_DWORD *)(v7 + 24) )
          break;
      }
      sub_16CCBA0(v7, v3);
      if ( v4 )
        goto LABEL_4;
LABEL_8:
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        return;
    }
    *(_DWORD *)(v7 + 28) = v10 + 1;
    *v9 = v3;
    ++*(_QWORD *)v7;
LABEL_4:
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)a1[1] + 24LL))(*(_QWORD *)a1[1], v3) )
    {
      v5 = a1[2];
      v6 = *(unsigned int *)(v5 + 8);
      if ( (unsigned int)v6 >= *(_DWORD *)(v5 + 12) )
      {
        sub_16CD150(a1[2], v5 + 16, 0, 8);
        v6 = *(unsigned int *)(v5 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v5 + 8 * v6) = v3;
      ++*(_DWORD *)(v5 + 8);
    }
    goto LABEL_8;
  }
}
