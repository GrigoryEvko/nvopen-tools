// Function: sub_2A4C850
// Address: 0x2a4c850
//
__int64 __fastcall sub_2A4C850(__int64 a1, char a2)
{
  __int64 v2; // r13
  __int64 v3; // r14
  _QWORD *v4; // r12
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r15
  char v10; // dl
  __int64 *v11; // r12
  char v12; // bl
  _QWORD *v13; // r13
  __int64 v14; // r15
  __int64 v15; // r12
  _QWORD *v16; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // [rsp+8h] [rbp-E8h]
  __int64 v22; // [rsp+10h] [rbp-E0h]
  __int64 v23; // [rsp+20h] [rbp-D0h]
  __int64 *v24; // [rsp+28h] [rbp-C8h]
  __int64 *v25; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v26; // [rsp+38h] [rbp-B8h]
  _BYTE v27[176]; // [rsp+40h] [rbp-B0h] BYREF

  v2 = *(_QWORD *)(a1 + 16);
  v25 = (__int64 *)v27;
  v26 = 0x1000000000LL;
  if ( !v2 )
    return 0;
  do
  {
    while ( 1 )
    {
      v3 = v2;
      v2 = *(_QWORD *)(v2 + 8);
      v4 = *(_QWORD **)(v3 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v4 - 61) > 1u )
        break;
LABEL_5:
      if ( !v2 )
        goto LABEL_19;
    }
    if ( sub_BD2BE0(*(_QWORD *)(v3 + 24)) )
    {
      sub_BD5D50(v3);
      goto LABEL_5;
    }
    if ( *(_BYTE *)(v4[1] + 8LL) == 7 )
    {
      if ( !a2
        || *(_BYTE *)v4 == 85
        && (v18 = *(v4 - 4)) != 0
        && !*(_BYTE *)v18
        && *(_QWORD *)(v18 + 24) == v4[10]
        && (*(_BYTE *)(v18 + 33) & 0x20) != 0 )
      {
        sub_B43D60(v4);
      }
      goto LABEL_5;
    }
    v7 = v4[2];
    v8 = (unsigned int)v26;
    while ( v7 )
    {
      v9 = v7;
      v7 = *(_QWORD *)(v7 + 8);
      v10 = **(_BYTE **)(v9 + 24);
      if ( (unsigned __int8)(v10 - 78) <= 1u || v10 == 63 )
      {
        if ( v8 + 1 > (unsigned __int64)HIDWORD(v26) )
        {
          sub_C8D5F0((__int64)&v25, v27, v8 + 1, 8u, v5, v6);
          v8 = (unsigned int)v26;
        }
        v25[v8] = v9;
        v8 = (unsigned int)(v26 + 1);
        LODWORD(v26) = v26 + 1;
      }
    }
    if ( v8 + 1 > (unsigned __int64)HIDWORD(v26) )
    {
      sub_C8D5F0((__int64)&v25, v27, v8 + 1, 8u, v5, v6);
      v8 = (unsigned int)v26;
    }
    v25[v8] = v3;
    LODWORD(v26) = v26 + 1;
  }
  while ( v2 );
LABEL_19:
  v11 = v25;
  v21 = &v25[(unsigned int)v26];
  if ( v21 == v25 )
  {
    v23 = 0;
  }
  else
  {
    v24 = v25;
    v12 = 1;
    v23 = 0;
    do
    {
      v13 = *(_QWORD **)(*v24 + 24);
      v22 = *v24;
      v14 = v13[2];
      while ( v14 )
      {
        v15 = v14;
        v14 = *(_QWORD *)(v14 + 8);
        v16 = *(_QWORD **)(v15 + 24);
        if ( *(_BYTE *)v16 == 85 )
        {
          v19 = *(v16 - 4);
          if ( v19 )
          {
            if ( !*(_BYTE *)v19 && *(_QWORD *)(v19 + 24) == v16[10] && (*(_BYTE *)(v19 + 33) & 0x20) != 0 )
            {
              if ( v12 && *(_DWORD *)(v19 + 36) == 211 )
              {
                if ( v23 )
                {
                  v23 = 0;
                  v12 = 0;
                }
                else
                {
                  v23 = v16[5];
                }
              }
              if ( sub_BD2BE0((__int64)v16) )
                sub_BD5D50(v15);
              else
                sub_B43D60(v16);
            }
          }
        }
      }
      if ( !a2
        || *(_BYTE *)v13 == 85
        && (v20 = *(v13 - 4)) != 0
        && !*(_BYTE *)v20
        && *(_QWORD *)(v20 + 24) == v13[10]
        && (*(_BYTE *)(v20 + 33) & 0x20) != 0 )
      {
        if ( sub_BD2BE0((__int64)v13) )
          sub_BD5D50(v22);
        else
          sub_B43D60(v13);
      }
      ++v24;
    }
    while ( v21 != v24 );
    v11 = v25;
  }
  if ( v11 != (__int64 *)v27 )
    _libc_free((unsigned __int64)v11);
  return v23;
}
