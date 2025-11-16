// Function: sub_1F3A960
// Address: 0x1f3a960
//
__int64 __fastcall sub_1F3A960(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  __int16 v4; // ax
  __int64 v5; // r15
  __int64 v6; // rdx
  unsigned int v7; // r8d
  __int64 v8; // r12
  __int64 v9; // r12
  __int64 v10; // rbx
  unsigned int v11; // r14d
  __int64 v12; // rdi
  char v13; // al
  int v15; // eax
  unsigned __int8 v16; // [rsp+7h] [rbp-39h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  v3 = a3;
  v4 = *(_WORD *)(a2 + 46);
  v5 = *(_QWORD *)(a2 + 16);
  if ( (v4 & 4) == 0 && (v4 & 8) != 0 )
  {
    LOBYTE(v15) = sub_1E15D00(a2, 0x40000u, 2);
    v3 = a3;
    LODWORD(v6) = v15;
  }
  else
  {
    v6 = (*(_QWORD *)(v5 + 8) >> 18) & 1LL;
  }
  v7 = 0;
  if ( (_BYTE)v6 )
  {
    v8 = *(unsigned int *)(a2 + 40);
    if ( (_DWORD)v8 )
    {
      v9 = 8 * v8;
      v10 = 0;
      v11 = 0;
      do
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v5 + 40) + v10 + 2) & 2) != 0 )
        {
          v12 = *(_QWORD *)(a2 + 32) + 5 * v10;
          v13 = *(_BYTE *)v12;
          if ( *(_BYTE *)v12 )
          {
            if ( v13 == 1 || v13 == 4 )
            {
              v7 = v6;
              *(_QWORD *)(v12 + 24) = *(_QWORD *)(v3 + 40LL * v11 + 24);
            }
          }
          else
          {
            v16 = v6;
            v17 = v3;
            sub_1E310D0(v12, *(_DWORD *)(v3 + 40LL * v11 + 8));
            LODWORD(v6) = v16;
            v3 = v17;
            v7 = v16;
          }
          ++v11;
        }
        v10 += 8;
      }
      while ( v9 != v10 );
    }
  }
  return v7;
}
