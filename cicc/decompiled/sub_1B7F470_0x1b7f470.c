// Function: sub_1B7F470
// Address: 0x1b7f470
//
void __fastcall sub_1B7F470(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 *v9; // rsi
  int v10; // edx
  __int64 *v11; // rcx
  __int64 v12; // rax
  __int64 **v13; // r14
  unsigned int v14; // r13d
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rbx
  unsigned __int64 v18; // rbx
  char v19; // al
  unsigned int v20; // r13d
  __int64 v21; // rax
  unsigned int v22; // esi
  int v23; // eax
  __int64 v24; // rsi
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  _QWORD *v27; // rax
  unsigned int v28; // ebx
  unsigned __int64 v29; // rcx
  __int64 v30; // [rsp+0h] [rbp-C0h]
  __int64 v31; // [rsp+10h] [rbp-B0h]
  __int64 v32; // [rsp+10h] [rbp-B0h]
  __int64 v33; // [rsp+10h] [rbp-B0h]
  char v35; // [rsp+27h] [rbp-99h]
  __int64 *v37; // [rsp+40h] [rbp-80h] BYREF
  __int64 v38; // [rsp+48h] [rbp-78h]
  _BYTE v39[112]; // [rsp+50h] [rbp-70h] BYREF

  v7 = a3;
  v8 = (8 * a3) >> 3;
  v37 = (__int64 *)v39;
  v38 = 0x800000000LL;
  if ( (unsigned __int64)(8 * a3) > 0x40 )
  {
    sub_16CD150((__int64)&v37, v39, (8 * a3) >> 3, 8, a5, a6);
    v9 = v37;
    v10 = v38;
    v11 = &v37[(unsigned int)v38];
  }
  else
  {
    v9 = (__int64 *)v39;
    v10 = 0;
    v11 = (__int64 *)v39;
  }
  if ( v7 > 0 )
  {
    v12 = 0;
    do
    {
      v11[v12] = (__int64)a2[v12];
      ++v12;
    }
    while ( v8 - v12 > 0 );
    v9 = v37;
    v10 = v38;
  }
  LODWORD(v38) = v8 + v10;
  sub_14C4AD0(a1, v9, v8 + v10);
  if ( *(_BYTE *)(a1 + 16) != 54 )
    goto LABEL_8;
  v35 = 0;
  v13 = &a2[v7 - 1];
  v14 = 0;
LABEL_12:
  v15 = 1;
  v16 = **v13;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v16 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v21 = *(_QWORD *)(v16 + 32);
        v16 = *(_QWORD *)(v16 + 24);
        v15 *= v21;
        continue;
      case 1:
        v17 = 16;
        goto LABEL_15;
      case 2:
        v17 = 32;
        goto LABEL_15;
      case 3:
      case 9:
        v17 = 64;
        goto LABEL_15;
      case 4:
        v17 = 80;
        goto LABEL_15;
      case 5:
      case 6:
        v17 = 128;
        goto LABEL_15;
      case 7:
        v31 = v15;
        v22 = 0;
        goto LABEL_23;
      case 0xB:
        v17 = *(_DWORD *)(v16 + 8) >> 8;
        goto LABEL_15;
      case 0xD:
        v33 = v15;
        v27 = (_QWORD *)sub_15A9930(a4, v16);
        v15 = v33;
        v17 = 8LL * *v27;
        goto LABEL_15;
      case 0xE:
        v30 = v15;
        v32 = *(_QWORD *)(v16 + 32);
        v24 = *(_QWORD *)(v16 + 24);
        v25 = (unsigned int)sub_15A9FE0(a4, v24);
        v26 = sub_127FA20(a4, v24);
        v15 = v30;
        v17 = 8 * v32 * v25 * ((v25 + ((unsigned __int64)(v26 + 7) >> 3) - 1) / v25);
        goto LABEL_15;
      case 0xF:
        v31 = v15;
        v22 = *(_DWORD *)(v16 + 8) >> 8;
LABEL_23:
        v23 = sub_15A9520(a4, v22);
        v15 = v31;
        v17 = (unsigned int)(8 * v23);
LABEL_15:
        v18 = (unsigned __int64)(v15 * v17) >> 3;
        v19 = sub_1C300D0();
        if ( !v19 )
        {
          v20 = ((1 << v18) - 1) | v14;
          if ( a2 == v13 )
            goto LABEL_30;
LABEL_17:
          --v13;
          v14 = v20 << v18;
          goto LABEL_12;
        }
        v35 = v19;
        v20 = sub_1C30110(*v13) | v14;
        if ( a2 != v13 )
          goto LABEL_17;
LABEL_30:
        v28 = (unsigned __int64)sub_127FA20(a4, *(_QWORD *)a1) >> 3;
        v29 = a3 * (unsigned int)((unsigned __int64)sub_127FA20(a4, **a2) >> 3);
        if ( v29 >= v28 )
        {
          if ( v35 )
            goto LABEL_33;
        }
        else
        {
          if ( v35 )
            v20 &= ~(-1 << v29);
          else
            v20 = (1 << v29) - 1;
LABEL_33:
          sub_1C30170(a1, v20);
        }
LABEL_8:
        if ( v37 != (__int64 *)v39 )
          _libc_free((unsigned __int64)v37);
        return;
    }
  }
}
