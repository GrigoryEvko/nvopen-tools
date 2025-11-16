// Function: sub_F82D10
// Address: 0xf82d10
//
void __fastcall sub_F82D10(__int64 a1, __int64 a2, __int64 a3, void *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 *v7; // rbx
  __int64 *v8; // r12
  __int64 v9; // r13
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int8 *v14; // rsi
  __int64 v15; // rax
  __int64 *v16; // [rsp+10h] [rbp-140h] BYREF
  __int64 v17; // [rsp+18h] [rbp-138h] BYREF
  __int64 v18; // [rsp+20h] [rbp-130h] BYREF
  unsigned __int8 *v19; // [rsp+28h] [rbp-128h]
  char v20; // [rsp+30h] [rbp-120h]
  _QWORD v21[35]; // [rsp+38h] [rbp-118h] BYREF

  if ( !*(_BYTE *)(a1 + 8) )
  {
    v6 = *(_QWORD *)a1;
    if ( *(_DWORD *)(*(_QWORD *)a1 + 304LL) )
    {
      v10 = *(_QWORD *)(v6 + 296);
      a3 = 48LL * *(unsigned int *)(v6 + 312);
      v11 = v10 + a3;
      if ( v10 != v10 + a3 )
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(v10 + 24);
          if ( v12 != -8192 && v12 != -4096 )
            break;
          v10 += 48;
          if ( v11 == v10 )
            goto LABEL_4;
        }
        if ( v11 != v10 )
        {
          do
          {
            v13 = *(_QWORD *)(v10 + 8);
            v18 = 0;
            v17 = v13 & 6;
            v19 = *(unsigned __int8 **)(v10 + 24);
            v14 = v19;
            if ( v19 + 4096 != 0 && v19 != 0 && v19 != (unsigned __int8 *)-8192LL )
            {
              sub_BD6050((unsigned __int64 *)&v17, *(_QWORD *)(v10 + 8) & 0xFFFFFFFFFFFFFFF8LL);
              v14 = v19;
            }
            v16 = (__int64 *)&unk_49E51C0;
            v20 = *(_BYTE *)(v10 + 32);
            v21[0] = *(_QWORD *)(v10 + 40);
            sub_F7D320((__int64)v21, v14);
            if ( !v20 )
            {
              a4 = &unk_49DB368;
              v16 = (__int64 *)&unk_49DB368;
              LOBYTE(a4) = v19 != 0;
              if ( v19 != 0 && v19 + 4096 != 0 && v19 != (unsigned __int8 *)-8192LL )
                sub_BD60C0(&v17);
            }
            v10 += 48;
            if ( v10 == v11 )
              break;
            while ( 1 )
            {
              v15 = *(_QWORD *)(v10 + 24);
              if ( v15 != -4096 && v15 != -8192 )
                break;
              v10 += 48;
              if ( v11 == v10 )
                goto LABEL_26;
            }
          }
          while ( v11 != v10 );
LABEL_26:
          v6 = *(_QWORD *)a1;
        }
      }
    }
LABEL_4:
    sub_F7CBB0((__int64)&v16, v6, a3, (__int64)a4, a5, a6);
    sub_F82360(*(_QWORD *)a1, v6);
    v7 = v16;
    v8 = &v16[(unsigned int)v17];
    if ( v16 != v8 )
    {
      do
      {
        v9 = *--v8;
        v6 = sub_ACADE0(*(__int64 ***)(v9 + 8));
        sub_BD84D0(v9, v6);
        sub_B43D60((_QWORD *)v9);
      }
      while ( v7 != v8 );
      v8 = v16;
    }
    if ( v8 != &v18 )
      _libc_free(v8, v6);
  }
}
