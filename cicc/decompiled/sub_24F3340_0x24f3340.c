// Function: sub_24F3340
// Address: 0x24f3340
//
void __fastcall sub_24F3340(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rdx
  _BYTE *v10; // rdi
  __int64 v11; // r13
  _BYTE *v12; // r14
  _QWORD **v13; // rbx
  _QWORD *v14; // r15
  __int64 *v15; // rax
  __int64 **v16; // rax
  __int64 v17; // rax
  _BYTE *v18; // [rsp+10h] [rbp-60h] BYREF
  __int64 v19; // [rsp+18h] [rbp-58h]
  _BYTE v20[80]; // [rsp+20h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  v18 = v20;
  v19 = 0x400000000LL;
  if ( v6 )
  {
    v7 = 0;
    do
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v6 + 24);
        if ( *(_BYTE *)v8 == 85 )
        {
          v9 = *(_QWORD *)(v8 - 32);
          if ( v9 )
          {
            if ( !*(_BYTE *)v9
              && *(_QWORD *)(v9 + 24) == *(_QWORD *)(v8 + 80)
              && (*(_BYTE *)(v9 + 33) & 0x20) != 0
              && *(_DWORD *)(v9 + 36) == 47 )
            {
              break;
            }
          }
        }
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          goto LABEL_13;
      }
      if ( v7 + 1 > (unsigned __int64)HIDWORD(v19) )
      {
        sub_C8D5F0((__int64)&v18, v20, v7 + 1, 8u, a5, a6);
        v7 = (unsigned int)v19;
      }
      *(_QWORD *)&v18[8 * v7] = v8;
      v7 = (unsigned int)(v19 + 1);
      LODWORD(v19) = v19 + 1;
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v6 );
LABEL_13:
    if ( (_DWORD)v7 )
    {
      if ( a2 )
      {
        v15 = (__int64 *)sub_BD5C60(a1);
        v16 = (__int64 **)sub_BCE3C0(v15, 0);
        v17 = sub_AC9EC0(v16);
        v10 = v18;
        v11 = v17;
        v7 = (unsigned int)v19;
      }
      else
      {
        v10 = v18;
        v11 = *(_QWORD *)(*(_QWORD *)v18 + 32 * (1LL - (*(_DWORD *)(*(_QWORD *)v18 + 4LL) & 0x7FFFFFF)));
      }
      v12 = &v10[8 * v7];
      if ( v12 == v10 )
        goto LABEL_20;
      v13 = (_QWORD **)v10;
      do
      {
        v14 = *v13++;
        sub_BD84D0((__int64)v14, v11);
        sub_B43D60(v14);
      }
      while ( v12 != (_BYTE *)v13 );
    }
    v10 = v18;
LABEL_20:
    if ( v10 != v20 )
      _libc_free((unsigned __int64)v10);
  }
}
