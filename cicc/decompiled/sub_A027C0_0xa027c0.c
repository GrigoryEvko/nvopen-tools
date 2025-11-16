// Function: sub_A027C0
// Address: 0xa027c0
//
void __fastcall sub_A027C0(__int64 a1)
{
  unsigned int i; // ebx
  __int64 v2; // rax
  unsigned __int8 v3; // dl
  __int64 v4; // rax
  _BYTE *v5; // r14
  unsigned __int8 v6; // cl
  unsigned int v7; // r15d
  bool v8; // dl
  __int64 v9; // rsi
  _BYTE *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 *v15; // r9
  __int64 *v16; // r13
  __int64 *v17; // r15
  __int64 v18; // rax
  __int64 v19; // rax
  _BYTE *v20; // r14
  __int64 v21; // rdx
  __int64 v22; // [rsp-70h] [rbp-70h]
  __int64 v23; // [rsp-70h] [rbp-70h]
  int v24; // [rsp-68h] [rbp-68h]
  __int64 v25; // [rsp-60h] [rbp-60h]
  __int64 *v26; // [rsp-58h] [rbp-58h] BYREF
  __int64 v27; // [rsp-50h] [rbp-50h]
  _BYTE v28[72]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_BYTE *)(a1 + 1098) )
  {
    v22 = sub_BA8DC0(*(_QWORD *)(a1 + 256), "llvm.dbg.cu", 11);
    if ( v22 )
    {
      v24 = sub_B91A00(v22);
      if ( v24 )
      {
        for ( i = 0; i != v24; ++i )
        {
          v2 = sub_B91A10(v22, i);
          v3 = *(_BYTE *)(v2 - 16);
          if ( (v3 & 2) != 0 )
            v4 = *(_QWORD *)(v2 - 32);
          else
            v4 = v2 - 16 - 8LL * ((v3 >> 2) & 0xF);
          v5 = *(_BYTE **)(v4 + 48);
          if ( v5 && *v5 == 5 )
          {
            v6 = *(v5 - 16);
            v7 = 0;
            v8 = (v6 & 2) != 0;
            while ( 1 )
            {
              if ( v8 )
              {
                if ( v7 >= *((_DWORD *)v5 - 6) )
                  break;
                v9 = *((_QWORD *)v5 - 4);
              }
              else
              {
                if ( v7 >= ((*((_WORD *)v5 - 8) >> 6) & 0xFu) )
                  break;
                v9 = (__int64)&v5[-8 * ((v6 >> 2) & 0xF) - 16];
              }
              v10 = *(_BYTE **)(v9 + 8LL * v7);
              if ( v10 )
              {
                if ( *v10 == 25 )
                {
                  v11 = sub_B0D000(*(_QWORD *)(a1 + 248), 0, 0, 0, 1);
                  v12 = sub_B0EF30(*(_QWORD *)(a1 + 248), v10, v11, 1, 1);
                  sub_BA6610(v5, v7, v12);
                  v6 = *(v5 - 16);
                  v8 = (v6 & 2) != 0;
                }
              }
              ++v7;
            }
          }
        }
      }
    }
    v13 = *(_QWORD *)(a1 + 256);
    v23 = v13 + 8;
    v25 = *(_QWORD *)(v13 + 16);
    if ( v13 + 8 != v25 )
    {
      do
      {
        v14 = v25 - 56;
        if ( !v25 )
          v14 = 0;
        v26 = (__int64 *)v28;
        v27 = 0x100000000LL;
        sub_B91D10(v14, 0, &v26);
        sub_B98000(v14, 0);
        v15 = v26;
        v16 = &v26[(unsigned int)v27];
        if ( v16 != v26 )
        {
          v17 = v26;
          do
          {
            while ( 1 )
            {
              v20 = (_BYTE *)*v17;
              if ( *(_BYTE *)*v17 != 25 )
                break;
              ++v17;
              v18 = sub_B0D000(*(_QWORD *)(a1 + 248), 0, 0, 0, 1);
              v19 = sub_B0EF30(*(_QWORD *)(a1 + 248), v20, v18, 1, 1);
              sub_B994D0(v14, 0, v19);
              if ( v16 == v17 )
                goto LABEL_28;
            }
            v21 = *v17++;
            sub_B994D0(v14, 0, v21);
          }
          while ( v16 != v17 );
LABEL_28:
          v15 = v26;
        }
        if ( v15 != (__int64 *)v28 )
          _libc_free(v15, 0);
        v25 = *(_QWORD *)(v25 + 8);
      }
      while ( v23 != v25 );
    }
  }
}
