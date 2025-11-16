// Function: sub_158BE00
// Address: 0x158be00
//
__int64 __fastcall sub_158BE00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  unsigned int v5; // eax
  unsigned int v7; // eax
  unsigned int v8; // eax
  __int64 v9; // r15
  __int64 v10; // rbx
  unsigned int v11; // eax
  unsigned int v12; // eax
  __int64 v13; // r14
  unsigned int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned int v17; // eax
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-38h]

  if ( sub_158A120(a2) || sub_158A0B0(a3) )
  {
    v4 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
      sub_16A4FD0(a1, a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    v5 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v5;
    if ( v5 <= 0x40 )
    {
LABEL_6:
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
      return a1;
    }
LABEL_56:
    sub_16A4FD0(a1 + 16, a2 + 16);
    return a1;
  }
  if ( sub_158A120(a3) || sub_158A0B0(a2) )
  {
    v7 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a1 + 8) = v7;
    if ( v7 > 0x40 )
      sub_16A4FD0(a1, a3);
    else
      *(_QWORD *)a1 = *(_QWORD *)a3;
    v8 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(a1 + 24) = v8;
    if ( v8 <= 0x40 )
    {
LABEL_15:
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
      return a1;
    }
LABEL_66:
    sub_16A4FD0(a1 + 16, a3 + 16);
    return a1;
  }
  if ( !sub_158A670(a2) && sub_158A670(a3) )
  {
    sub_158BE00(a1, a3, a2);
    return a1;
  }
  v9 = a2 + 16;
  if ( !sub_158A670(a2) && !sub_158A670(a3) )
  {
    if ( (int)sub_16A9900(a2, a3) >= 0 )
    {
      v10 = a3 + 16;
      if ( (int)sub_16A9900(a2 + 16, a3 + 16) >= 0 )
      {
        if ( (int)sub_16A9900(a2, a3 + 16) >= 0 )
        {
LABEL_25:
          sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
          return a1;
        }
        goto LABEL_57;
      }
LABEL_68:
      sub_1455FD0(a1, a2);
      return a1;
    }
    if ( (int)sub_16A9900(a2 + 16, a3) <= 0 )
      goto LABEL_25;
    if ( (int)sub_16A9900(v9, a3 + 16) >= 0 )
    {
      v16 = *(_DWORD *)(a3 + 8);
      *(_DWORD *)(a1 + 8) = v16;
      if ( v16 > 0x40 )
        sub_16A4FD0(a1, a3);
      else
        *(_QWORD *)a1 = *(_QWORD *)a3;
      v17 = *(_DWORD *)(a3 + 24);
      *(_DWORD *)(a1 + 24) = v17;
      if ( v17 <= 0x40 )
        goto LABEL_15;
      goto LABEL_66;
    }
LABEL_73:
    sub_13A38D0((__int64)&v20, v9);
    sub_13A38D0((__int64)&v18, a3);
    sub_15898E0(a1, (__int64)&v18, &v20);
    sub_135E100(&v18);
    sub_135E100(&v20);
    return a1;
  }
  if ( sub_158A670(a2) && !sub_158A670(a3) )
  {
    if ( (int)sub_16A9900(a3, a2 + 16) >= 0 )
    {
      if ( (int)sub_16A9900(a3, a2) < 0 )
      {
        v13 = a3 + 16;
        if ( (int)sub_16A9900(v13, a2) > 0 )
        {
          sub_13A38D0((__int64)&v20, v13);
          sub_13A38D0((__int64)&v18, a2);
          sub_15898E0(a1, (__int64)&v18, &v20);
          sub_135E100(&v18);
          sub_135E100(&v20);
          return a1;
        }
        goto LABEL_25;
      }
      goto LABEL_69;
    }
    if ( (int)sub_16A9900(a3 + 16, a2 + 16) < 0 )
      goto LABEL_69;
    if ( (int)sub_16A9900(a3 + 16, a2) <= 0 )
      goto LABEL_73;
LABEL_67:
    if ( sub_158A690(a2, a3) )
      goto LABEL_68;
LABEL_69:
    sub_1455FD0(a1, a3);
    return a1;
  }
  v10 = a3 + 16;
  if ( (int)sub_16A9900(a3 + 16, a2 + 16) >= 0 )
  {
    if ( (int)sub_16A9900(a3 + 16, a2) <= 0 )
    {
      if ( (int)sub_16A9900(a3, a2) >= 0 )
      {
        v21 = *(_DWORD *)(a2 + 24);
        if ( v21 > 0x40 )
          sub_16A4FD0(&v20, a2 + 16);
        else
          v20 = *(_QWORD *)(a2 + 16);
        v19 = *(_DWORD *)(a3 + 8);
        if ( v19 > 0x40 )
          sub_16A4FD0(&v18, a3);
        else
          v18 = *(_QWORD *)a3;
        goto LABEL_34;
      }
LABEL_53:
      v14 = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 8) = v14;
      if ( v14 > 0x40 )
        sub_16A4FD0(a1, a2);
      else
        *(_QWORD *)a1 = *(_QWORD *)a2;
      v15 = *(_DWORD *)(a2 + 24);
      *(_DWORD *)(a1 + 24) = v15;
      if ( v15 <= 0x40 )
        goto LABEL_6;
      goto LABEL_56;
    }
    if ( sub_158A690(a2, a3) )
      goto LABEL_53;
LABEL_42:
    v11 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a1 + 8) = v11;
    if ( v11 > 0x40 )
      sub_16A4FD0(a1, a3);
    else
      *(_QWORD *)a1 = *(_QWORD *)a3;
    v12 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(a1 + 24) = v12;
    if ( v12 <= 0x40 )
      goto LABEL_15;
    goto LABEL_66;
  }
  if ( (int)sub_16A9900(a3, a2 + 16) < 0 )
    goto LABEL_67;
  if ( (int)sub_16A9900(a3, a2) >= 0 )
    goto LABEL_42;
LABEL_57:
  v21 = *(_DWORD *)(a3 + 24);
  if ( v21 > 0x40 )
    sub_16A4FD0(&v20, v10);
  else
    v20 = *(_QWORD *)(a3 + 16);
  v19 = *(_DWORD *)(a2 + 8);
  if ( v19 > 0x40 )
    sub_16A4FD0(&v18, a2);
  else
    v18 = *(_QWORD *)a2;
LABEL_34:
  sub_15898E0(a1, (__int64)&v18, &v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  return a1;
}
