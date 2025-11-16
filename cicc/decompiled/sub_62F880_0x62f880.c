// Function: sub_62F880
// Address: 0x62f880
//
void __fastcall sub_62F880(__int64 a1, __int64 a2)
{
  __int64 i; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned int v7; // eax
  __int64 v8; // rdx
  int v9; // ecx
  __int64 v10; // r14
  __int64 v11; // rsi
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __int64 j; // rax
  __int64 v15; // rdx
  __int64 v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+8h] [rbp-38h]
  __int64 v18; // [rsp+8h] [rbp-38h]

  for ( i = *(_QWORD *)(a1 + 128); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(i + 169) & 2) != 0 )
  {
    v4 = *(_QWORD *)(a1 + 176);
    v5 = sub_7259C0(8);
    v6 = *(_QWORD *)(i + 160);
    v16 = v5;
    v7 = sub_8D4070(v6);
    v8 = v16;
    v9 = v7;
    if ( v7 )
    {
      if ( v4 )
      {
        v9 = 1;
        v6 = 0;
        goto LABEL_6;
      }
      *(_QWORD *)(a1 + 128) = v6;
      sub_62F880(a1, a2, v16, v7);
      v6 = *(_QWORD *)(a1 + 128);
      v8 = v16;
      *(_QWORD *)(a1 + 128) = 0;
    }
    else if ( v4 )
    {
LABEL_6:
      v10 = 0;
      while ( 1 )
      {
        if ( *(_BYTE *)(v4 + 173) == 13 )
          goto LABEL_8;
        ++v10;
        if ( !v9 )
          goto LABEL_8;
        v11 = *(_QWORD *)(v4 + 128);
        if ( v6 )
        {
          v12 = *(_QWORD *)(v4 + 128);
          if ( *(_BYTE *)(v11 + 140) == 12 )
          {
            do
              v12 = *(_QWORD *)(v12 + 160);
            while ( *(_BYTE *)(v12 + 140) == 12 );
          }
          v13 = *(_QWORD *)(v12 + 176);
          for ( j = v6; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          v4 = *(_QWORD *)(v4 + 120);
          if ( v13 > *(_QWORD *)(j + 176) )
            v6 = v11;
          if ( !v4 )
          {
LABEL_19:
            sub_73C230(i, v16);
            v15 = v16;
            *(_BYTE *)(v16 + 169) &= 0xECu;
            *(_QWORD *)(v16 + 160) = v6;
            *(_QWORD *)(v16 + 176) = v10;
            if ( v10 )
              goto LABEL_20;
            goto LABEL_25;
          }
        }
        else
        {
          v6 = *(_QWORD *)(v4 + 128);
LABEL_8:
          v4 = *(_QWORD *)(v4 + 120);
          if ( !v4 )
            goto LABEL_19;
        }
      }
    }
    v18 = v8;
    sub_73C230(i, v8);
    v15 = v18;
    *(_BYTE *)(v18 + 169) &= 0xECu;
    *(_QWORD *)(v18 + 160) = v6;
    *(_QWORD *)(v18 + 176) = 0;
LABEL_25:
    *(_BYTE *)(v15 + 169) |= 0x20u;
LABEL_20:
    *(_QWORD *)(v15 + 128) = 0;
    v17 = v15;
    sub_8D6090(v15);
    *(_QWORD *)(a1 + 128) = v17;
  }
}
