// Function: sub_D22BA0
// Address: 0xd22ba0
//
__int64 __fastcall sub_D22BA0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rax
  char v6; // al
  __int64 *v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // rax
  char v13; // dl
  __int64 i; // [rsp+0h] [rbp-60h] BYREF
  __int64 *v15; // [rsp+8h] [rbp-58h]
  __int64 v16; // [rsp+10h] [rbp-50h]
  int v17; // [rsp+18h] [rbp-48h]
  char v18; // [rsp+1Ch] [rbp-44h]
  __int64 v19; // [rsp+20h] [rbp-40h] BYREF

  v2 = sub_D22B80(a1);
  if ( !(_BYTE)v2 )
    return v2;
  v3 = *(_QWORD *)(a1 - 64);
  v18 = 1;
  v15 = &v19;
  v16 = 0x100000002LL;
  v17 = 0;
  v19 = v3;
  for ( i = 1; ; ++i )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 + 56);
      if ( v3 + 48 != v4 )
      {
        while ( 1 )
        {
          if ( !v4 )
            BUG();
          if ( *(_BYTE *)(v4 - 24) == 85 )
          {
            v5 = *(_QWORD *)(v4 - 56);
            if ( v5 )
            {
              if ( !*(_BYTE *)v5 )
              {
                a2 = *(_QWORD *)(v4 + 56);
                if ( *(_QWORD *)(v5 + 24) == a2 && *(_DWORD *)(v5 + 36) == 146 )
                  break;
              }
            }
          }
          if ( (unsigned __int8)sub_B46970((unsigned __int8 *)(v4 - 24)) )
            goto LABEL_24;
          v4 = *(_QWORD *)(v4 + 8);
          if ( v3 + 48 == v4 )
            goto LABEL_17;
        }
        v6 = v18;
        goto LABEL_14;
      }
LABEL_17:
      v3 = sub_AA5780(v3);
      if ( !v3 )
      {
LABEL_24:
        v6 = v18;
        v2 = 0;
        goto LABEL_14;
      }
      if ( v18 )
        break;
LABEL_25:
      a2 = v3;
      sub_C8CC70((__int64)&i, v3, (__int64)v8, v9, v10, v11);
      v6 = v18;
      if ( !v13 )
      {
        v2 = 0;
LABEL_14:
        if ( !v6 )
          _libc_free(v15, a2);
        return v2;
      }
    }
    v12 = v15;
    v9 = HIDWORD(v16);
    v8 = &v15[HIDWORD(v16)];
    if ( v15 != v8 )
      break;
LABEL_27:
    if ( HIDWORD(v16) >= (unsigned int)v16 )
      goto LABEL_25;
    ++HIDWORD(v16);
    *v8 = v3;
  }
  while ( v3 != *v12 )
  {
    if ( v8 == ++v12 )
      goto LABEL_27;
  }
  return 0;
}
