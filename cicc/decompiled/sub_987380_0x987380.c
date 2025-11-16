// Function: sub_987380
// Address: 0x987380
//
bool __fastcall sub_987380(char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // ebx
  char v6; // al
  char *v7; // r12
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  char v13; // r13
  _QWORD *i; // r14
  int v15; // [rsp+Ch] [rbp-64h]
  __int64 v16; // [rsp+10h] [rbp-60h]
  __int64 v17; // [rsp+20h] [rbp-50h] BYREF
  __int64 v18; // [rsp+28h] [rbp-48h]
  char v19; // [rsp+34h] [rbp-3Ch]

  v5 = a2 & 2;
  if ( (a2 & 2) != 0 )
    return 1;
  v6 = *a1;
  if ( *a1 != 18 )
  {
    if ( v6 != 16 )
      return v6 == 14;
    v9 = *(_BYTE *)(sub_AC5230() + 8);
    if ( v9 > 3u && v9 != 5 && (v9 & 0xFD) != 4 )
      return 0;
    v15 = sub_AC5290(a1);
    if ( v15 )
    {
      v16 = sub_C33340(a1, a2, v10, v11, v12);
      while ( 1 )
      {
        sub_AC5470(&v17, a1, v5);
        if ( v17 == v16 )
        {
          v13 = *(_BYTE *)(v18 + 20) & 7;
          for ( i = (_QWORD *)(v18 + 24LL * *(_QWORD *)(v18 - 8)); (_QWORD *)v18 != i; sub_91D830(i) )
            i -= 3;
          j_j_j___libc_free_0_0(i - 1);
        }
        else
        {
          v13 = v19 & 7;
          sub_C338F0(&v17);
        }
        if ( v13 == 1 )
          break;
        if ( v15 == ++v5 )
          return 1;
      }
      return 0;
    }
    return 1;
  }
  if ( *((_QWORD *)a1 + 3) == sub_C33340(a1, a2, a3, a4, a5) )
    v7 = (char *)*((_QWORD *)a1 + 4);
  else
    v7 = a1 + 24;
  return (v7[20] & 7) != 1;
}
