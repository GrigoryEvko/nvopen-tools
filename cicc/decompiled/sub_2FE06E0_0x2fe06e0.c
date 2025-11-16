// Function: sub_2FE06E0
// Address: 0x2fe06e0
//
unsigned __int64 __fastcall sub_2FE06E0(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  char v5; // cl
  char v6; // r14
  __int64 v8; // rdx
  __int64 (*v9)(); // rax
  __int64 (*v10)(); // rax
  __int64 v11; // r13
  __int64 v12; // rdx
  char v13; // al
  char v15; // al
  __int64 (*v16)(); // rax
  char v17; // al
  unsigned int v18; // eax
  unsigned int v19; // et0
  char v20; // [rsp+Fh] [rbp-41h]
  char v21; // [rsp+Fh] [rbp-41h]
  unsigned int v22; // [rsp+18h] [rbp-38h]

  v5 = 0;
  v6 = 0;
  v8 = *a1;
  v9 = *(__int64 (**)())(*a1 + 640);
  if ( v9 != sub_2FDC5C0 )
  {
    v15 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD, _QWORD))v9)(a1, a3, 0, 0);
    v8 = *a1;
    v5 = v15;
    v16 = *(__int64 (**)())(*a1 + 640);
    if ( v16 != sub_2FDC5C0 )
    {
      v20 = v5;
      v17 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD))v16)(a1, a4, 0);
      v5 = v20;
      v6 = v17;
      if ( v20 && v17 )
        return ((unsigned __int64)*(unsigned __int16 *)(a3 + 68) << 32) | *(unsigned __int16 *)(a3 + 68);
      v8 = *a1;
    }
  }
  v10 = *(__int64 (**)())(v8 + 648);
  v11 = *(unsigned __int16 *)(a3 + 68);
  if ( v10 == sub_2FDC5D0 )
  {
    v12 = v22;
  }
  else
  {
    v21 = v5;
    v18 = ((__int64 (__fastcall *)(__int64 *, _QWORD))v10)(a1, (unsigned int)v11);
    v5 = v21;
    v12 = v18;
  }
  v13 = v5 ^ 1;
  if ( !v5 )
  {
    if ( a2 == 2 )
    {
      if ( !v13 )
      {
        if ( !v6 )
          goto LABEL_10;
LABEL_48:
        BUG();
      }
      if ( !v6 )
        goto LABEL_10;
      return (v11 << 32) | (unsigned int)v12;
    }
    if ( a2 <= 2 )
    {
      if ( a2 )
      {
        if ( v13 )
        {
          if ( !v6 )
            return (v12 << 32) | (unsigned int)v11;
LABEL_10:
          v12 = v11;
          return (v12 << 32) | (unsigned int)v11;
        }
        goto LABEL_41;
      }
      if ( !v6 )
        return (v12 << 32) | (unsigned int)v11;
      return (v11 << 32) | (unsigned int)v12;
    }
    if ( a2 == 3 )
    {
      if ( v13 )
      {
        if ( v6 )
          goto LABEL_10;
        return v12 | (v11 << 32);
      }
LABEL_23:
      if ( v6 )
        goto LABEL_48;
      return v12 | (v11 << 32);
    }
LABEL_47:
    BUG();
  }
  if ( a2 == 2 )
    goto LABEL_23;
  if ( a2 > 2 )
  {
    if ( a2 == 3 )
      goto LABEL_23;
    goto LABEL_47;
  }
  if ( a2 )
  {
    if ( v13 && v6 )
    {
      v11 = (unsigned int)v12;
      goto LABEL_10;
    }
    if ( v6 != 1 )
      return (v12 << 32) | (unsigned int)v11;
    v19 = v11;
    LODWORD(v11) = v12;
    v12 = v19;
LABEL_41:
    if ( !v6 && !v5 )
      return (v12 << 32) | (unsigned int)v11;
    goto LABEL_48;
  }
  if ( v6 )
    goto LABEL_48;
  return (v12 << 32) | (unsigned int)v12;
}
