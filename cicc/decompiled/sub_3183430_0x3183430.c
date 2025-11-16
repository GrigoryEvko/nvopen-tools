// Function: sub_3183430
// Address: 0x3183430
//
__int64 __fastcall sub_3183430(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // r15
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r14
  __int16 v14; // dx
  unsigned __int64 v15; // rsi
  char v16; // al
  char v17; // dl
  __int64 v18; // rax
  unsigned __int8 v19; // r14
  char v20; // al
  unsigned int v21; // edx
  char v24; // [rsp+1Eh] [rbp-A2h]
  char v25; // [rsp+1Fh] [rbp-A1h]
  __int64 v26; // [rsp+20h] [rbp-A0h]
  __int64 v27; // [rsp+28h] [rbp-98h]
  _QWORD v28[4]; // [rsp+30h] [rbp-90h] BYREF
  int v29; // [rsp+50h] [rbp-70h]
  char v30; // [rsp+54h] [rbp-6Ch]
  void *v31; // [rsp+60h] [rbp-60h] BYREF
  __int16 v32; // [rsp+80h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 80);
  v26 = a2 + 72;
  if ( v4 == a2 + 72 )
  {
    v20 = 0;
    v19 = 0;
  }
  else
  {
    v24 = 0;
    v25 = 0;
    do
    {
      if ( !v4 )
        BUG();
      v5 = *(_QWORD *)(v4 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v5 == v4 + 24 )
        goto LABEL_28;
      if ( !v5 )
        BUG();
      v6 = v5 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
LABEL_28:
        BUG();
      if ( *(_BYTE *)(v5 - 24) == 34
        && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v5 + 56) + 16LL) + 8LL) != 7
        && *(char *)(v5 - 17) < 0 )
      {
        v7 = sub_BD2BC0(v5 - 24);
        v9 = v7 + v8;
        if ( *(char *)(v5 - 17) < 0 )
          v9 -= sub_BD2BC0(v5 - 24);
        v10 = v9 >> 4;
        if ( (_DWORD)v10 )
        {
          v27 = 16LL * (unsigned int)v10;
          v11 = 0;
          while ( 1 )
          {
            v12 = 0;
            if ( *(char *)(v5 - 17) < 0 )
              v12 = sub_BD2BC0(v5 - 24);
            if ( *(_DWORD *)(*(_QWORD *)(v12 + v11) + 8LL) == 6 )
              break;
            v11 += 16;
            if ( v11 == v27 )
              goto LABEL_23;
          }
          v13 = *(_QWORD *)(v5 - 120);
          if ( !sub_AA54C0(v13) )
          {
            v32 = 257;
            memset(&v28[1], 0, 24);
            v28[0] = a3;
            v29 = 0;
            v30 = 1;
            v24 = 1;
            v13 = sub_F451F0(v6, 0, (__int64)v28, &v31);
          }
          v15 = sub_AA5190(v13);
          if ( v15 )
          {
            v16 = v14;
            v17 = HIBYTE(v14);
          }
          else
          {
            v17 = 0;
            v16 = 0;
          }
          LOBYTE(v3) = v16;
          v18 = v3;
          BYTE1(v18) = v17;
          v3 = v18;
          sub_3183360(a1, v15, v18, v6);
          v25 = 1;
        }
      }
LABEL_23:
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v26 != v4 );
    v19 = v25;
    v20 = v24;
  }
  v21 = v19;
  BYTE1(v21) = v20;
  return v21;
}
