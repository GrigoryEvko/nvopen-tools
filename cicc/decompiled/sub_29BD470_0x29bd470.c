// Function: sub_29BD470
// Address: 0x29bd470
//
__int64 __fastcall sub_29BD470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v8; // rdx
  unsigned int i; // eax
  __int64 v10; // r14
  unsigned __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  char v16; // al
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  char v20; // al
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  char v24; // al
  char v25; // al
  unsigned __int64 v27; // [rsp+0h] [rbp-90h]
  int v28; // [rsp+Ch] [rbp-84h]
  char *v30; // [rsp+20h] [rbp-70h] BYREF
  __int64 v31; // [rsp+28h] [rbp-68h]
  _BYTE v32[96]; // [rsp+30h] [rbp-60h] BYREF

  v30 = v32;
  v31 = 0x600000000LL;
  v28 = 0;
  if ( a3 == a2 )
  {
    *(_QWORD *)(a1 + 8) = 0x600000000LL;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 64) = 1;
  }
  else
  {
    v5 = a2;
    if ( !a2 )
      goto LABEL_17;
LABEL_3:
    v8 = (unsigned int)(*(_DWORD *)(v5 + 44) + 1);
    for ( i = *(_DWORD *)(v5 + 44) + 1; ; i = 0 )
    {
      if ( i >= *(_DWORD *)(a4 + 32) )
        BUG();
      v10 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v8) + 8LL);
      v11 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v11 == v10 + 48 )
        goto LABEL_27;
      if ( !v11 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
LABEL_27:
        BUG();
      if ( *(_BYTE *)(v11 - 24) != 31 )
        goto LABEL_18;
      v27 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      sub_B19AA0(a5, v5, **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v8) + 8LL));
      if ( !v16 )
      {
        sub_B19AA0(a5, v5, *(_QWORD *)(v27 - 56));
        if ( v20 )
        {
          v25 = sub_29BD190((__int64)&v30, *(_QWORD *)(v27 - 120) | 4LL, v17, v27, v18, v19);
        }
        else
        {
          sub_B19AA0(a5, v5, *(_QWORD *)(v27 - 88));
          if ( !v24 )
            goto LABEL_18;
          v25 = sub_29BD190((__int64)&v30, *(_QWORD *)(v27 - 120) & 0xFFFFFFFFFFFFFFFBLL, v21, v27, v22, v23);
        }
        if ( v25 )
        {
          if ( ++v28 == 7 )
          {
LABEL_18:
            *(_BYTE *)(a1 + 64) = 0;
            goto LABEL_19;
          }
        }
      }
      if ( a3 == v10 )
        break;
      v5 = v10;
      if ( v10 )
        goto LABEL_3;
LABEL_17:
      v8 = 0;
    }
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x600000000LL;
    if ( (_DWORD)v31 )
      sub_29BD310(a1, &v30, v12, v13, v14, v15);
    *(_BYTE *)(a1 + 64) = 1;
LABEL_19:
    if ( v30 != v32 )
      _libc_free((unsigned __int64)v30);
  }
  return a1;
}
