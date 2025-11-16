// Function: sub_AD7F90
// Address: 0xad7f90
//
_BOOL8 __fastcall sub_AD7F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v5; // r12
  __int64 v6; // rax
  _BYTE *v7; // rdi
  __int64 v8; // rbx
  char v9; // al
  char v10; // al
  __int64 v12; // rdx
  int v13; // eax
  int v14; // r14d
  unsigned int v15; // r13d
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  _BYTE *v20; // rbx
  _BYTE *v21; // rdi
  __int64 v22; // r15
  char v24; // al
  unsigned __int8 *v25; // rax

  v5 = (unsigned __int8 *)a1;
  if ( *(_BYTE *)a1 != 18 )
  {
    v12 = *(_QWORD *)(a1 + 8);
    v13 = *(unsigned __int8 *)(v12 + 8);
    if ( (_BYTE)v13 == 17 )
    {
      v14 = *(_DWORD *)(v12 + 32);
      if ( v14 )
      {
        v15 = 0;
        while ( 1 )
        {
          v16 = (_BYTE *)sub_AD69F0(v5, v15);
          v20 = v16;
          if ( !v16 || *v16 != 18 )
            break;
          v21 = v16 + 24;
          v22 = sub_C33340(v5, v15, v17, v18, v19);
          if ( *((_QWORD *)v20 + 3) == v22 ? sub_C40310(v21) : (unsigned __int8)sub_C33940(v21) )
            break;
          if ( v22 == *((_QWORD *)v20 + 3) )
          {
            v24 = *(_BYTE *)(*((_QWORD *)v20 + 4) + 20LL) & 7;
            if ( v24 == 1 )
              return 0;
          }
          else
          {
            v24 = v20[44] & 7;
            if ( v24 == 1 )
              return 0;
          }
          if ( v24 == 3 || !v24 )
            break;
          if ( v14 == ++v15 )
            return 1;
        }
        return 0;
      }
      return 1;
    }
    if ( (unsigned int)(v13 - 17) > 1 )
      return 0;
    a2 = 0;
    v25 = sub_AD7630(a1, 0, v12);
    v5 = v25;
    if ( !v25 || *v25 != 18 )
      return 0;
  }
  v6 = sub_C33340(a1, a2, a3, a4, a5);
  v7 = v5 + 24;
  v8 = v6;
  if ( *((_QWORD *)v5 + 3) == v6 )
    v9 = sub_C40310(v7);
  else
    v9 = sub_C33940(v7);
  if ( v9 )
    return 0;
  if ( v8 == *((_QWORD *)v5 + 3) )
  {
    v10 = *(_BYTE *)(*((_QWORD *)v5 + 4) + 20LL) & 7;
    if ( v10 != 1 )
      return v10 != 3 && v10;
    return 0;
  }
  v10 = v5[44] & 7;
  if ( v10 == 1 )
    return 0;
  return v10 != 3 && v10;
}
