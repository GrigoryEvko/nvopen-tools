// Function: sub_7F7570
// Address: 0x7f7570
//
__int64 __fastcall sub_7F7570(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int8 v3; // dl
  unsigned int v4; // r14d
  __int64 v6; // r13
  __int64 v7; // r13
  char v8; // al
  __int64 v9; // rax
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-80h] BYREF
  __int64 v14; // [rsp+8h] [rbp-78h]
  int v15; // [rsp+10h] [rbp-70h]
  int v16; // [rsp+14h] [rbp-6Ch]
  unsigned int v17; // [rsp+20h] [rbp-60h] BYREF
  __int64 v18; // [rsp+28h] [rbp-58h]
  unsigned __int64 v19; // [rsp+30h] [rbp-50h]
  __int64 v20; // [rsp+38h] [rbp-48h]
  __int64 v21; // [rsp+40h] [rbp-40h]
  int v22; // [rsp+48h] [rbp-38h]

  v2 = a2;
  if ( *(_BYTE *)(a1 + 173) == 2 )
  {
    if ( *(_BYTE *)(a2 + 140) == 12 )
    {
      do
        v2 = *(_QWORD *)(v2 + 160);
      while ( *(_BYTE *)(v2 + 140) == 12 );
    }
    v3 = *(_QWORD *)(a1 + 176) < *(_QWORD *)(v2 + 176);
    v4 = v3;
    goto LABEL_5;
  }
  if ( (unsigned int)sub_8D4070(a2) || (v4 = sub_8D23E0(a2)) != 0 )
  {
LABEL_7:
    v3 = 1;
    v4 = 1;
    goto LABEL_5;
  }
  v6 = *(_QWORD *)(a1 + 176);
  sub_7F5D80(a2, 11, (__int64)&v17);
  sub_7F51D0(v6, 1, 1, (__int64)&v13);
  v7 = v13;
  if ( v13 )
  {
    while ( 1 )
    {
      v8 = *(_BYTE *)(v7 + 173);
      if ( v8 == 13 )
      {
        v11 = *(_QWORD *)(v7 + 184);
        if ( (*(_BYTE *)(v7 + 176) & 1) != 0 )
        {
          v18 = *(_QWORD *)(v7 + 184);
          v20 = *(_QWORD *)(v11 + 120);
        }
        else
        {
          v19 = *(_QWORD *)(v7 + 184);
        }
        v7 = *(_QWORD *)(v7 + 120);
        sub_7F51D0(v7, 1, 1, (__int64)&v13);
        v8 = *(_BYTE *)(v7 + 173);
      }
      if ( v8 == 11 )
        v7 = *(_QWORD *)(v7 + 176);
      if ( (unsigned int)sub_8D3B80(v20)
        && *(_BYTE *)(v7 + 173) != 9
        && !(unsigned int)sub_7E1F90(v20)
        && (unsigned int)sub_7F7570(v7, *(_QWORD *)(v7 + 128)) )
      {
        goto LABEL_7;
      }
      if ( v14 )
      {
        v9 = v19 + v14;
        v14 = 0;
        v19 = v9 - 1;
      }
      if ( v13 )
      {
        sub_7F51D0(*(_QWORD *)(v13 + 120), v15, v16, (__int64)&v13);
        v7 = v13;
        if ( v13 )
        {
          if ( v17 )
          {
            ++v19;
          }
          else
          {
            v12 = sub_72FD90(*(_QWORD *)(v18 + 112), v22);
            v7 = v13;
            v18 = v12;
            v20 = *(_QWORD *)(v12 + 120);
          }
          if ( v7 )
            continue;
        }
      }
      goto LABEL_20;
    }
  }
  if ( v17 )
  {
    if ( v19 <= v21 - 1 )
      goto LABEL_7;
  }
  else if ( v18 )
  {
    goto LABEL_7;
  }
LABEL_20:
  v10 = sub_8D3B10(a2);
  v3 = 0;
  if ( !v10 )
  {
    v4 = v17;
    if ( v17 )
    {
      v3 = v19 < v21 - 1;
      v4 = v3;
    }
    else if ( v18 )
    {
      v3 = sub_72FD90(*(_QWORD *)(v18 + 112), 11) != 0;
      v4 = v3;
    }
  }
LABEL_5:
  *(_BYTE *)(a1 + 170) = (v3 << 6) | (32 * v3) | *(_BYTE *)(a1 + 170) & 0x9F;
  return v4;
}
