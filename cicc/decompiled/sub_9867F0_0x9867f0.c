// Function: sub_9867F0
// Address: 0x9867f0
//
__int64 __fastcall sub_9867F0(unsigned int a1, char *a2)
{
  unsigned __int8 v2; // al
  _BYTE *v3; // rdx
  unsigned int v4; // r12d
  _BYTE *v6; // rax
  unsigned int v7; // r12d
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned int v10; // ebx
  int v11; // r14d
  unsigned int v12; // r13d
  unsigned int v13; // r15d
  _BYTE *v14; // rax
  unsigned int v15; // eax
  int v16; // [rsp+4h] [rbp-7Ch]
  __int64 v17; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-68h]
  __int64 v19; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-58h]
  __int64 v21; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-48h]
  __int64 v23; // [rsp+40h] [rbp-40h]
  unsigned int v24; // [rsp+48h] [rbp-38h]

  if ( a1 == 34 )
    return 1;
  if ( a1 != 33 )
  {
    v18 = sub_BCB060(*((_QWORD *)a2 + 1));
    if ( v18 > 0x40 )
      sub_C43690(&v17, 0, 0);
    else
      v17 = 0;
    v2 = *a2;
    v3 = a2 + 24;
    if ( *a2 == 17 )
      goto LABEL_6;
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a2 + 1) + 8LL) - 17 <= 1 )
    {
      v4 = 0;
      if ( v2 > 0x15u )
      {
LABEL_10:
        if ( v18 > 0x40 && v17 )
          j_j___libc_free_0_0(v17);
        return v4;
      }
      v6 = (_BYTE *)sub_AD7630(a2, 0);
      if ( v6 )
      {
        v3 = v6 + 24;
        if ( *v6 == 17 )
        {
LABEL_6:
          sub_AB1A50(&v21, a1, v3);
          v4 = sub_AB1B10(&v21, &v17) ^ 1;
          if ( v24 > 0x40 && v23 )
            j_j___libc_free_0_0(v23);
          if ( v22 > 0x40 && v21 )
            j_j___libc_free_0_0(v21);
          goto LABEL_10;
        }
      }
      v2 = *a2;
    }
    v4 = 0;
    if ( v2 == 16 )
    {
      v16 = sub_AC5290(a2);
      if ( v16 )
      {
        v7 = 0;
        while ( 1 )
        {
          sub_AC5390(&v19, a2, v7);
          sub_AB1A50(&v21, a1, &v19);
          if ( v20 > 0x40 && v19 )
            j_j___libc_free_0_0(v19);
          if ( (unsigned __int8)sub_AB1B10(&v21, &v17) )
            break;
          if ( v24 > 0x40 && v23 )
            j_j___libc_free_0_0(v23);
          if ( v22 > 0x40 && v21 )
            j_j___libc_free_0_0(v21);
          if ( v16 == ++v7 )
            goto LABEL_49;
        }
        if ( v24 > 0x40 && v23 )
          j_j___libc_free_0_0(v23);
        if ( v22 > 0x40 && v21 )
          j_j___libc_free_0_0(v21);
        v4 = 0;
      }
      else
      {
LABEL_49:
        v4 = 1;
      }
    }
    goto LABEL_10;
  }
  v4 = 0;
  if ( (unsigned __int8)*a2 > 0x15u )
    return v4;
  v4 = sub_AC30F0(a2);
  if ( (_BYTE)v4 )
    return 1;
  if ( *a2 == 17 )
  {
    v4 = *((_DWORD *)a2 + 8);
    if ( v4 <= 0x40 )
      LOBYTE(v4) = *((_QWORD *)a2 + 3) == 0;
    else
      LOBYTE(v4) = v4 == (unsigned int)sub_C444A0(a2 + 24);
  }
  else
  {
    v8 = *((_QWORD *)a2 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    {
      v9 = sub_AD7630(a2, 0);
      if ( v9 && *(_BYTE *)v9 == 17 )
      {
        v10 = *(_DWORD *)(v9 + 32);
        if ( v10 <= 0x40 )
          LOBYTE(v4) = *(_QWORD *)(v9 + 24) == 0;
        else
          LOBYTE(v4) = v10 == (unsigned int)sub_C444A0(v9 + 24);
      }
      else if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v11 = *(_DWORD *)(v8 + 32);
        if ( v11 )
        {
          v12 = 0;
          v13 = 0;
          while ( 1 )
          {
            v14 = (_BYTE *)sub_AD69F0(a2, v13);
            if ( !v14 )
              break;
            if ( *v14 != 13 )
            {
              if ( *v14 != 17 )
                break;
              LOBYTE(v15) = sub_9867B0((__int64)(v14 + 24));
              v12 = v15;
              if ( !(_BYTE)v15 )
                break;
            }
            if ( v11 == ++v13 )
              return v12;
          }
        }
      }
    }
  }
  return v4;
}
