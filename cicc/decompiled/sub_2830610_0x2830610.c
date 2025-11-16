// Function: sub_2830610
// Address: 0x2830610
//
__int64 __fastcall sub_2830610(_QWORD *a1, unsigned __int8 **a2)
{
  _QWORD *v2; // r12
  unsigned __int8 *v3; // rbx
  unsigned __int8 **v4; // rax
  __int64 v5; // rdx
  unsigned __int8 **v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned __int8 **v9; // rdx
  int v10; // eax
  __int64 v12; // rax
  unsigned __int8 *v13; // rcx
  __int64 v14; // rdx
  bool v15; // zf
  __int64 v16; // rax
  unsigned __int8 *v17; // rbx
  __int64 v18; // rax
  unsigned __int8 *v19; // rbx
  __int64 v20; // [rsp+8h] [rbp-28h] BYREF
  __int64 v21; // [rsp+10h] [rbp-20h] BYREF
  __int64 v22[3]; // [rsp+18h] [rbp-18h] BYREF

  v2 = a1;
  v3 = *a2;
  v4 = *(unsigned __int8 ***)(*a1 + 96LL);
  v5 = 8LL * *(unsigned int *)(*a1 + 104LL);
  v6 = &v4[(unsigned __int64)v5 / 8];
  v7 = v5 >> 3;
  v8 = v5 >> 5;
  if ( !v8 )
  {
LABEL_16:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          goto LABEL_9;
        goto LABEL_19;
      }
      if ( v3 == *v4 )
        goto LABEL_8;
      ++v4;
    }
    if ( v3 == *v4 )
      goto LABEL_8;
    ++v4;
LABEL_19:
    if ( v3 != *v4 )
      goto LABEL_9;
    goto LABEL_8;
  }
  v9 = &v4[4 * v8];
  while ( 1 )
  {
    if ( v3 == *v4 )
      goto LABEL_8;
    if ( v3 == v4[1] )
    {
      ++v4;
      goto LABEL_8;
    }
    if ( v3 == v4[2] )
    {
      v4 += 2;
      goto LABEL_8;
    }
    if ( v3 == v4[3] )
      break;
    v4 += 4;
    if ( v4 == v9 )
    {
      v7 = v6 - v4;
      goto LABEL_16;
    }
  }
  v4 += 3;
LABEL_8:
  if ( v6 != v4 )
    return 1;
LABEL_9:
  v10 = *v3;
  if ( (unsigned __int8)v10 <= 0x15u )
    return 1;
  if ( (unsigned __int8)v10 <= 0x1Cu )
    return 0;
  if ( (unsigned int)(v10 - 67) > 0xC )
  {
    if ( (unsigned int)(v10 - 42) > 0x11 )
      return 0;
    v12 = a1[1];
    if ( (v3[7] & 0x40) != 0 )
      v13 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
    else
      v13 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
    v14 = *(_QWORD *)v13;
    v15 = *(_QWORD *)(v12 + 16) == 0;
    v21 = *(_QWORD *)v13;
    if ( !v15 )
    {
      v6 = (unsigned __int8 **)&v21;
      a1 = (_QWORD *)v12;
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64 *))(v12 + 24))(v12, &v21) )
        return 0;
      v16 = v2[1];
      v17 = (v3[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v3 - 1) : &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
      v14 = *((_QWORD *)v17 + 4);
      v15 = *(_QWORD *)(v16 + 16) == 0;
      v22[0] = v14;
      if ( !v15 )
        return (*(__int64 (__fastcall **)(__int64, __int64 *))(v16 + 24))(v16, v22);
    }
LABEL_43:
    sub_4263D6(a1, v6, v14);
  }
  v18 = a1[1];
  if ( (v3[7] & 0x40) != 0 )
    v19 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
  else
    v19 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
  v14 = *(_QWORD *)v19;
  v15 = *(_QWORD *)(v18 + 16) == 0;
  v20 = *(_QWORD *)v19;
  if ( v15 )
    goto LABEL_43;
  return (*(__int64 (__fastcall **)(__int64, __int64 *, __int64, __int64))(v18 + 24))(v18, &v20, v14, v7);
}
