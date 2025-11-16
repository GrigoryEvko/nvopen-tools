// Function: sub_987210
// Address: 0x987210
//
bool __fastcall sub_987210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned int v11; // r13d
  char v12; // r12
  _QWORD *i; // r14
  int v14; // [rsp+Ch] [rbp-64h]
  __int64 v15; // [rsp+10h] [rbp-60h]
  __int64 v16; // [rsp+20h] [rbp-50h] BYREF
  __int64 v17; // [rsp+28h] [rbp-48h]
  char v18; // [rsp+34h] [rbp-3Ch]

  if ( *(_BYTE *)a1 != 18 )
  {
    if ( *(_BYTE *)a1 == 16 )
    {
      v7 = *(_BYTE *)(sub_AC5230() + 8);
      if ( v7 <= 3u || v7 == 5 || (v7 & 0xFD) == 4 )
      {
        v14 = sub_AC5290(a1);
        if ( !v14 )
          return 1;
        v11 = 0;
        v15 = sub_C33340(a1, a2, v8, v9, v10);
        while ( 1 )
        {
          sub_AC5470(&v16, a1, v11);
          if ( v15 == v16 )
          {
            v12 = *(_BYTE *)(v17 + 20) & 7;
            for ( i = (_QWORD *)(v17 + 24LL * *(_QWORD *)(v17 - 8)); (_QWORD *)v17 != i; sub_91D830(i) )
              i -= 3;
            j_j_j___libc_free_0_0(i - 1);
          }
          else
          {
            v12 = v18 & 7;
            sub_C338F0(&v16);
          }
          if ( v12 == 3 )
            break;
          if ( v14 == ++v11 )
            return 1;
        }
      }
    }
    return 0;
  }
  if ( *(_QWORD *)(a1 + 24) == sub_C33340(a1, a2, a3, a4, a5) )
    v5 = *(_QWORD *)(a1 + 32);
  else
    v5 = a1 + 24;
  return (*(_BYTE *)(v5 + 20) & 7) != 3;
}
