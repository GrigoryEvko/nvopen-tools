// Function: sub_390C520
// Address: 0x390c520
//
__int64 __fastcall sub_390C520(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned int v4; // r12d
  char v5; // r8
  __int64 v6; // rbx
  unsigned int v7; // edx
  unsigned int v8; // r14d
  char v9; // si
  char *v10; // rax
  unsigned int v11; // ebx
  char *v12; // rax
  char *v13; // rax
  char v15; // r14
  char *v16; // rax
  char v17; // si
  unsigned int v18; // ecx
  char v19; // al
  __int64 v20; // r14
  char v21; // bl
  char v22; // al
  unsigned int v23; // r14d
  char *v24; // rax
  char *v25; // rax
  char v26; // [rsp+0h] [rbp-80h]
  char v27; // [rsp+7h] [rbp-79h]
  unsigned int v28; // [rsp+8h] [rbp-78h]
  unsigned int v29; // [rsp+8h] [rbp-78h]
  char v30; // [rsp+Ch] [rbp-74h]
  __int64 v31; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v32[2]; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 v33; // [rsp+30h] [rbp-50h]
  char *v34; // [rsp+38h] [rbp-48h]
  int v35; // [rsp+40h] [rbp-40h]
  __int64 v36; // [rsp+48h] [rbp-38h]

  v4 = *(_DWORD *)(a3 + 72);
  if ( !sub_38CF260(*(_QWORD *)(a3 + 48), &v31, a2) )
    sub_16BD130("sleb128 and uleb128 expressions must be absolute", 1u);
  *(_DWORD *)(a3 + 72) = 0;
  v35 = 1;
  v32[0] = &unk_49EFC48;
  v34 = 0;
  v33 = 0;
  v32[1] = 0;
  v36 = a3 + 64;
  sub_16E7A40((__int64)v32, 0, 0, 0);
  v5 = *(_BYTE *)(a3 + 56);
  v6 = v31;
  v7 = 1;
  if ( !v5 )
  {
    v8 = 0;
    do
    {
      while ( 1 )
      {
        ++v8;
        v9 = v6 & 0x7F;
        v6 = (unsigned __int64)v6 >> 7;
        if ( v6 || v4 > v8 )
          v9 |= 0x80u;
        v10 = v34;
        if ( (unsigned __int64)v34 >= v33 )
          break;
        ++v34;
        *v10 = v9;
        if ( !v6 )
          goto LABEL_9;
      }
      sub_16E7DE0((__int64)v32, v9);
    }
    while ( v6 );
LABEL_9:
    if ( v4 > v8 )
    {
      v11 = v4 - 1;
      if ( v8 < v4 - 1 )
      {
        do
        {
          while ( 1 )
          {
            v12 = v34;
            if ( (unsigned __int64)v34 >= v33 )
              break;
            ++v8;
            ++v34;
            *v12 = 0x80;
            if ( v8 == v11 )
              goto LABEL_15;
          }
          ++v8;
          sub_16E7DE0((__int64)v32, 128);
        }
        while ( v8 != v11 );
      }
LABEL_15:
      v13 = v34;
      if ( (unsigned __int64)v34 >= v33 )
      {
        sub_16E7DE0((__int64)v32, 0);
      }
      else
      {
        ++v34;
        *v13 = 0;
      }
    }
    goto LABEL_17;
  }
  while ( 1 )
  {
    v19 = v6;
    v17 = v6 & 0x7F;
    v6 >>= 7;
    if ( v6 )
    {
      if ( v6 != -1 || (v19 & 0x40) == 0 )
      {
        v15 = v5;
        goto LABEL_19;
      }
    }
    else
    {
      v15 = v5;
      if ( (v19 & 0x40) != 0 )
        goto LABEL_19;
    }
    v15 = 0;
    if ( v4 <= v7 )
      break;
LABEL_19:
    v16 = v34;
    v17 |= 0x80u;
    if ( (unsigned __int64)v34 >= v33 )
      goto LABEL_28;
LABEL_20:
    v34 = v16 + 1;
    v18 = v7 + 1;
    *v16 = v17;
    if ( !v15 )
      goto LABEL_29;
LABEL_21:
    v7 = v18;
  }
  v16 = v34;
  if ( (unsigned __int64)v34 < v33 )
    goto LABEL_20;
LABEL_28:
  v28 = v7;
  v30 = v5;
  sub_16E7DE0((__int64)v32, v17);
  v7 = v28;
  v5 = v30;
  v18 = v28 + 1;
  if ( v15 )
    goto LABEL_21;
LABEL_29:
  if ( v4 > v7 )
  {
    v20 = v6 >> 63;
    v21 = (v6 >> 63) | 0x80;
    v27 = v20 & 0x7F;
    v22 = v20;
    v23 = v4 - 1;
    v26 = v22 & 0x7F;
    if ( v4 - 1 > v7 )
    {
      while ( 1 )
      {
        v24 = v34;
        if ( (unsigned __int64)v34 < v33 )
        {
          ++v34;
          *v24 = v21;
          if ( v18 == v23 )
            break;
        }
        else
        {
          v29 = v18;
          sub_16E7DE0((__int64)v32, v21);
          v18 = v29;
          if ( v29 == v23 )
            break;
        }
        ++v18;
      }
    }
    v25 = v34;
    if ( (unsigned __int64)v34 >= v33 )
    {
      sub_16E7DE0((__int64)v32, v26);
    }
    else
    {
      ++v34;
      *v25 = v27;
    }
  }
LABEL_17:
  LOBYTE(v4) = *(_DWORD *)(a3 + 72) != v4;
  v32[0] = &unk_49EFD28;
  sub_16E7960((__int64)v32);
  return v4;
}
