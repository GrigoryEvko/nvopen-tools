// Function: sub_1633E30
// Address: 0x1633e30
//
__int64 __fastcall sub_1633E30(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rdx
  const char *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // r12
  _QWORD *v9; // rbx
  __int64 v10; // rax
  _QWORD *v11; // r14
  __int64 v12; // rsi
  __int64 *v13; // rax
  __int64 *v14; // rdi
  unsigned int v15; // r8d
  __int64 *v16; // rcx

  v4 = (-(__int64)(a3 == 0) & 0xFFFFFFFFFFFFFFF7LL) + 18;
  v6 = "llvm.compiler.used";
  if ( !a3 )
    v6 = "llvm.used";
  v7 = sub_16321C0(a1, (__int64)v6, v4, 0);
  v8 = v7;
  if ( v7 && !sub_15E4F60(v7) )
  {
    v9 = *(_QWORD **)(v8 - 24);
    v10 = 24LL * (*((_DWORD *)v9 + 5) & 0xFFFFFFF);
    if ( (*((_BYTE *)v9 + 23) & 0x40) != 0 )
    {
      v11 = (_QWORD *)*(v9 - 1);
      v9 = &v11[(unsigned __int64)v10 / 8];
    }
    else
    {
      v11 = &v9[v10 / 0xFFFFFFFFFFFFFFF8LL];
    }
    for ( ; v9 != v11; v11 += 3 )
    {
LABEL_11:
      v12 = sub_1649F00(*v11);
      v13 = *(__int64 **)(a2 + 8);
      if ( *(__int64 **)(a2 + 16) != v13 )
        goto LABEL_9;
      v14 = &v13[*(unsigned int *)(a2 + 28)];
      v15 = *(_DWORD *)(a2 + 28);
      if ( v13 != v14 )
      {
        v16 = 0;
        while ( v12 != *v13 )
        {
          if ( *v13 == -2 )
            v16 = v13;
          if ( v14 == ++v13 )
          {
            if ( !v16 )
              goto LABEL_21;
            v11 += 3;
            *v16 = v12;
            --*(_DWORD *)(a2 + 32);
            ++*(_QWORD *)a2;
            if ( v9 != v11 )
              goto LABEL_11;
            return v8;
          }
        }
        continue;
      }
LABEL_21:
      if ( v15 < *(_DWORD *)(a2 + 24) )
      {
        *(_DWORD *)(a2 + 28) = v15 + 1;
        *v14 = v12;
        ++*(_QWORD *)a2;
      }
      else
      {
LABEL_9:
        sub_16CCBA0(a2, v12);
      }
    }
  }
  return v8;
}
