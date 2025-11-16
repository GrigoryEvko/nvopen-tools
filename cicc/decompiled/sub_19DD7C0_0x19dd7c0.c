// Function: sub_19DD7C0
// Address: 0x19dd7c0
//
__int64 __fastcall sub_19DD7C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  int v7; // r9d
  unsigned int v8; // edx
  __int64 *v9; // r13
  __int64 v10; // rdi
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rax

  v3 = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 56);
    v7 = 1;
    v8 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v4 + 72LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_3:
      if ( v9 != (__int64 *)(v4 + 72 * v3) )
      {
        while ( 1 )
        {
          v11 = *((_DWORD *)v9 + 4);
          if ( !v11 )
            break;
          while ( 1 )
          {
            v12 = v9[1];
            v13 = *(_QWORD *)(v12 + 24LL * v11 - 8);
            if ( v13 )
            {
              if ( sub_15CCEE0(*(_QWORD *)(a1 + 16), v13, a3) )
                return v13;
              v12 = v9[1];
              v11 = *((_DWORD *)v9 + 4);
            }
            v14 = v11 - 1;
            *((_DWORD *)v9 + 4) = v14;
            v15 = (_QWORD *)(v12 + 24 * v14);
            v16 = v15[2];
            if ( v16 == 0 || v16 == -8 || v16 == -16 )
              break;
            sub_1649B30(v15);
            v11 = *((_DWORD *)v9 + 4);
            if ( !v11 )
              return 0;
          }
        }
      }
    }
    else
    {
      while ( v10 != -8 )
      {
        v8 = (v3 - 1) & (v7 + v8);
        v9 = (__int64 *)(v4 + 72LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_3;
        ++v7;
      }
    }
  }
  return 0;
}
