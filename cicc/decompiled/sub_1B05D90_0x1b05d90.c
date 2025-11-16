// Function: sub_1B05D90
// Address: 0x1b05d90
//
void __fastcall sub_1B05D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v10; // rdx
  __int64 v11; // r10
  __int64 v12; // r15
  __int64 v13; // rax
  char v14; // di
  unsigned int v15; // esi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 i; // rax
  int v23; // r9d
  __int64 *v24; // r15
  __int64 v25; // rsi
  _BYTE *v26; // rsi
  __int64 v27; // rax
  __int64 *v28; // r8
  __int64 *v29; // r10
  __int64 v30; // r15
  __int64 *v31; // [rsp+0h] [rbp-B0h]
  __int64 *v32; // [rsp+8h] [rbp-A8h]
  __int64 v33; // [rsp+8h] [rbp-A8h]
  __int64 v34; // [rsp+10h] [rbp-A0h]
  __int64 v35; // [rsp+18h] [rbp-98h]
  __int64 *v36; // [rsp+28h] [rbp-88h] BYREF
  _BYTE *v37; // [rsp+30h] [rbp-80h] BYREF
  __int64 v38; // [rsp+38h] [rbp-78h]
  _BYTE v39[112]; // [rsp+40h] [rbp-70h] BYREF

  v37 = v39;
  v38 = 0x800000000LL;
  v8 = sub_157F280(a1);
  v11 = v10;
  v12 = v8;
  while ( v11 != v12 )
  {
    v13 = 0x17FFFFFFE8LL;
    v14 = *(_BYTE *)(v12 + 23) & 0x40;
    v15 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
    if ( v15 )
    {
      v16 = 24LL * *(unsigned int *)(v12 + 56) + 8;
      v17 = 0;
      do
      {
        v18 = v12 - 24LL * v15;
        if ( v14 )
          v18 = *(_QWORD *)(v12 - 8);
        if ( a2 == *(_QWORD *)(v18 + v16) )
        {
          v13 = 24 * v17;
          goto LABEL_9;
        }
        ++v17;
        v16 += 8;
      }
      while ( v15 != (_DWORD)v17 );
      v13 = 0x17FFFFFFE8LL;
    }
LABEL_9:
    if ( v14 )
    {
      v19 = *(_QWORD *)(*(_QWORD *)(v12 - 8) + v13);
      if ( !v19 )
        goto LABEL_38;
    }
    else
    {
      v19 = *(_QWORD *)(v12 - 24LL * v15 + v13);
      if ( !v19 )
LABEL_38:
        BUG();
    }
    if ( *(_BYTE *)(v19 + 16) > 0x17u )
    {
      v20 = (unsigned int)v38;
      if ( (unsigned int)v38 >= HIDWORD(v38) )
      {
        v33 = v11;
        v34 = a2;
        v35 = v19;
        sub_16CD150((__int64)&v37, v39, 0, 8, -24, a2);
        v20 = (unsigned int)v38;
        v11 = v33;
        a2 = v34;
        v19 = v35;
      }
      *(_QWORD *)&v37[8 * v20] = v19;
      LODWORD(v38) = v38 + 1;
    }
    v21 = *(_QWORD *)(v12 + 32);
    if ( !v21 )
      BUG();
    v12 = 0;
    if ( *(_BYTE *)(v21 - 8) == 77 )
      v12 = v21 - 24;
  }
  LODWORD(i) = v38;
  while ( (_DWORD)i )
  {
    v24 = *(__int64 **)&v37[8 * (unsigned int)i - 8];
    LODWORD(v38) = i - 1;
    v25 = v24[5];
    v36 = v24;
    if ( !sub_183E920(a5, v25) )
      goto LABEL_20;
    v26 = *(_BYTE **)(a4 + 8);
    if ( v26 == *(_BYTE **)(a4 + 16) )
    {
      sub_170B610(a4, v26, &v36);
LABEL_20:
      if ( sub_183E920(a3, v24[5]) )
        goto LABEL_28;
LABEL_21:
      LODWORD(i) = v38;
    }
    else
    {
      if ( v26 )
      {
        *(_QWORD *)v26 = v36;
        v26 = *(_BYTE **)(a4 + 8);
      }
      *(_QWORD *)(a4 + 8) = v26 + 8;
      if ( !sub_183E920(a3, v24[5]) )
        goto LABEL_21;
LABEL_28:
      v27 = 24LL * (*((_DWORD *)v24 + 5) & 0xFFFFFFF);
      if ( (*((_BYTE *)v24 + 23) & 0x40) != 0 )
      {
        v28 = (__int64 *)*(v24 - 1);
        v29 = &v28[(unsigned __int64)v27 / 8];
      }
      else
      {
        v29 = v24;
        v28 = &v24[v27 / 0xFFFFFFFFFFFFFFF8LL];
      }
      for ( i = (unsigned int)v38; v29 != v28; v28 += 3 )
      {
        v30 = *v28;
        if ( *(_BYTE *)(*v28 + 16) > 0x17u )
        {
          if ( (unsigned int)i >= HIDWORD(v38) )
          {
            v31 = v28;
            v32 = v29;
            sub_16CD150((__int64)&v37, v39, 0, 8, (int)v28, v23);
            i = (unsigned int)v38;
            v28 = v31;
            v29 = v32;
          }
          *(_QWORD *)&v37[8 * i] = v30;
          i = (unsigned int)(v38 + 1);
          LODWORD(v38) = v38 + 1;
        }
      }
    }
  }
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
}
